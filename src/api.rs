use std::pin::pin;
use std::sync::{Arc, OnceLock};

use aha::models::{
    EmbedModel, EmbeddingParameters, GenerateModel, ModelInstance, RerankModel, RerankParameters,
    WhichModel, load_model, qwen3_embedding::model::Qwen3EmbeddingModel,
    qwen3_reranker::model::Qwen3RerankerModel,
};
use aha::utils::{error_utils::{api_json_result, internal_error, ApiError}, string_to_static_str};
use anyhow::Context;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use rocket::futures::StreamExt;
use rocket::serde::json::Json;
use rocket::{
    Request,
    futures::Stream,
    http::{ContentType, Status},
    post,
    response::{Responder, stream::TextStream},
};
use tokio::sync::RwLock;

static MODEL: OnceLock<Arc<RwLock<ModelInstance<'static>>>> = OnceLock::new();
static EMBEDDING_MODEL: OnceLock<Arc<RwLock<Qwen3EmbeddingModel>>> = OnceLock::new();
static RERANK_MODEL: OnceLock<Arc<RwLock<Qwen3RerankerModel>>> = OnceLock::new();

/// Helper function to get the model reference with proper error handling
fn get_model_ref() -> anyhow::Result<Arc<RwLock<ModelInstance<'static>>>> {
    MODEL
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Model not initialized"))
        .with_context(|| "Failed to get model reference")
}

/// Helper function to get the embedding model reference with proper error handling
fn get_embedding_model_ref() -> anyhow::Result<Arc<RwLock<Qwen3EmbeddingModel>>> {
    EMBEDDING_MODEL
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Embedding model not initialized"))
        .with_context(|| "Failed to get embedding model reference")
}

/// Helper function to get the reranker model reference with proper error handling
fn get_rerank_model_ref() -> anyhow::Result<Arc<RwLock<Qwen3RerankerModel>>> {
    RERANK_MODEL
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Reranker model not initialized"))
        .with_context(|| "Failed to get reranker model reference")
}

pub fn init(model_type: WhichModel, path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = load_model(model_type, model_path)?;
    MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub fn init_embedding(path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = Qwen3EmbeddingModel::init(model_path, None, None)?;
    EMBEDDING_MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub fn init_rerank(path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = Qwen3RerankerModel::init(model_path, None, None)?;
    RERANK_MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub(crate) enum Response<R: Stream<Item = String> + Send> {
    Stream(TextStream<R>),
    Text(String),
    Error(String),
}

impl<'r, 'o: 'r, R> Responder<'r, 'o> for Response<R>
where
    R: Stream<Item = String> + Send + 'o,
    'r: 'o,
{
    fn respond_to(self, req: &'r Request<'_>) -> rocket::response::Result<'o> {
        match self {
            Response::Stream(stream) => stream.respond_to(req),
            Response::Text(text) => text.respond_to(req),
            Response::Error(e) => {
                let mut res = rocket::response::Response::new();
                res.set_status(Status::InternalServerError);
                res.set_header(ContentType::JSON);
                res.set_sized_body(e.len(), std::io::Cursor::new(e));
                Ok(res)
            }
        }
    }
}

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(
    req: Json<ChatCompletionParameters>,
) -> (ContentType, Response<impl Stream<Item = String> + Send>) {
    match req.stream {
        Some(false) => {
            let response = async {
                let model_ref = get_model_ref()?;
                let mut guard = model_ref.write().await;
                guard.generate(req.into_inner())
                    .with_context(|| "Failed to generate response")
            }.await;

            match response {
                Ok(res) => {
                    match serde_json::to_string(&res) {
                        Ok(response_str) => (ContentType::Text, Response::Text(response_str)),
                        Err(e) => {
                            let api_error = ApiError::internal(e.to_string());
                            match serde_json::to_string(&api_error) {
                                Ok(error_json) => (ContentType::JSON, Response::Error(error_json)),
                                Err(serialize_err) => (ContentType::Text, Response::Error(serialize_err.to_string())),
                            }
                        }
                    }
                }
                Err(e) => {
                    let api_error = ApiError::internal(e.to_string());
                    match serde_json::to_string(&api_error) {
                        Ok(error_json) => (ContentType::JSON, Response::Error(error_json)),
                        Err(serialize_err) => (ContentType::Text, Response::Error(serialize_err.to_string())),
                    }
                }
            }
        }
        _ => {
            let text_stream = TextStream! {
                match get_model_ref() {
                    Ok(model_ref) => {
                        let mut guard = model_ref.write().await;
                        match guard.generate_stream(req.into_inner()) {
                            Ok(stream) => {
                                let mut stream = pin!(stream);
                                while let Some(result) = stream.next().await {
                                    match result {
                                        Ok(chunk) => {
                                            match serde_json::to_string(&chunk) {
                                                Ok(json_str) => yield format!("data: {}\n\n", json_str),
                                                Err(e) => {
                                                    let api_error = ApiError::internal(e.to_string());
                                                    match serde_json::to_string(&api_error) {
                                                        Ok(error_json) => yield format!("data: {}\n\n", error_json),
                                                        Err(serialize_err) => yield format!("data: {{\"error\": \"{}\"}}\n\n", serialize_err),
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let api_error = ApiError::internal(e.to_string());
                                            match serde_json::to_string(&api_error) {
                                                Ok(error_json) => yield format!("data: {}\n\n", error_json),
                                                Err(serialize_err) => yield format!("data: {{\"error\": \"{}\"}}\n\n", serialize_err),
                                            }
                                            break;
                                        }
                                    }
                                }
                                yield "data: [DONE]\n\n".to_string();
                            },
                            Err(e) => {
                                let api_error = ApiError::internal(e.to_string());
                                match serde_json::to_string(&api_error) {
                                    Ok(error_json) => yield format!("event: error\ndata: {}\n\n", error_json),
                                    Err(serialize_err) => yield format!("event: error\ndata: {}\n\n", serialize_err),
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let api_error = ApiError::internal(e.to_string());
                        match serde_json::to_string(&api_error) {
                            Ok(error_json) => yield format!("event: error\ndata: {}\n\n", error_json),
                            Err(serialize_err) => yield format!("event: error\ndata: {}\n\n", serialize_err),
                        }
                    }
                }
            };
            (ContentType::EventStream, Response::Stream(text_stream))
        }
    }
}

#[post("/remove_background", data = "<req>")]
pub(crate) async fn remove_background(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = async {
        let model_ref = get_model_ref()?;
        let mut guard = model_ref.write().await;
        guard.generate(req.into_inner())
            .with_context(|| "Failed to remove background")
    }.await;

    match response {
        Ok(res) => api_json_result(res),
        Err(e) => internal_error(e),
    }
}

#[post("/speech", data = "<req>")]
pub(crate) async fn speech(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = async {
        let model_ref = get_model_ref()?;
        let mut guard = model_ref.write().await;
        guard.generate(req.into_inner())
            .with_context(|| "Failed to generate speech")
    }.await;

    match response {
        Ok(res) => api_json_result(res),
        Err(e) => internal_error(e),
    }
}

#[post("/embeddings", data = "<req>")]
pub(crate) async fn embeddings(req: Json<EmbeddingParameters>) -> (Status, String) {
    let response = async {
        let model_ref = get_embedding_model_ref()?;
        let mut guard = model_ref.write().await;
        guard.embed(req.into_inner())
            .with_context(|| "Failed to generate embeddings")
    }.await;

    match response {
        Ok(res) => api_json_result(res),
        Err(e) => internal_error(e),
    }
}

#[post("/rerank", data = "<req>")]
pub(crate) async fn rerank(req: Json<RerankParameters>) -> (Status, String) {
    let response = async {
        let model_ref = get_rerank_model_ref()?;
        let mut guard = model_ref.write().await;
        guard.rerank(req.into_inner())
            .with_context(|| "Failed to rerank documents")
    }.await;

    match response {
        Ok(res) => api_json_result(res),
        Err(e) => internal_error(e),
    }
}
