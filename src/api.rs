use std::pin::pin;
use std::sync::{Arc, OnceLock};

use aha::models::{
    EmbedModel, EmbeddingParameters, GenerateModel, ModelInstance, RerankModel, RerankParameters,
    WhichModel, load_model, qwen3_embedding::model::Qwen3EmbeddingModel,
    qwen3_reranker::model::Qwen3RerankerModel,
};
use aha::utils::string_to_static_str;
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
            let response = {
                let model_ref = MODEL
                    .get()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model not init"))
                    .unwrap();
                model_ref.write().await.generate(req.into_inner())
            };
            match response {
                Ok(res) => {
                    let response_str = serde_json::to_string(&res).unwrap();
                    (ContentType::Text, Response::Text(response_str))
                }
                Err(e) => (ContentType::Text, Response::Error(e.to_string())),
            }
        }
        _ => {
            let text_stream = TextStream! {
                let model_ref = MODEL.get().cloned().ok_or_else(|| anyhow::anyhow!("model not init")).unwrap();
                let mut guard = model_ref.write().await;
                let stream_result = guard.generate_stream(req.into_inner());
                match stream_result {
                    Ok(stream) => {
                        let mut stream = pin!(stream);
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(chunk) => {
                                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                                        yield format!("data: {}\n\n", json_str);
                                    }
                                }
                                Err(e) => {
                                    yield format!("data: {{\"error\": \"{}\"}}\n\n", e);
                                    break;
                                }
                            }
                        }
                        yield "data: [DONE]\n\n".to_string();
                    },
                    Err(e) => {
                        yield format!("event: error\ndata: {}\n\n", e.to_string());
                    }
                }
            };
            (ContentType::EventStream, Response::Stream(text_stream))
        }
    }
}

#[post("/remove_background", data = "<req>")]
pub(crate) async fn remove_background(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        model_ref.write().await.generate(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}

#[post("/speech", data = "<req>")]
pub(crate) async fn speech(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        model_ref.write().await.generate(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}

#[post("/embeddings", data = "<req>")]
pub(crate) async fn embeddings(req: Json<EmbeddingParameters>) -> (Status, String) {
    let response = {
        let model_ref = EMBEDDING_MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("embedding model not init"))
            .unwrap();
        model_ref.write().await.embed(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}

#[post("/rerank", data = "<req>")]
pub(crate) async fn rerank(req: Json<RerankParameters>) -> (Status, String) {
    let response = {
        let model_ref = RERANK_MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("rerank model not init"))
            .unwrap();
        model_ref.write().await.rerank(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}
