use axum::{Json as AxumJson};
use axum::response::IntoResponse;
use serde_json::Value;

use crate::{
    params::embedding::{EmbeddingData, EmbeddingRequest, EmbeddingResponse},
    server::api::MODEL,
};

fn parse_embedding_input(input: &Value) -> anyhow::Result<Vec<String>> {
    match input {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or_else(|| {
                    anyhow::anyhow!("embedding input array must contain only strings")
                })?;
                out.push(s.to_string());
            }
            if out.is_empty() {
                return Err(anyhow::anyhow!("embedding input cannot be empty"));
            }
            Ok(out)
        }
        _ => Err(anyhow::anyhow!(
            "embedding input must be a string or an array of strings"
        )),
    }
}

pub(crate) async fn embeddings(
    AxumJson(req): AxumJson<EmbeddingRequest>,
) -> axum::response::Response {
    let texts = match parse_embedding_input(&req.input) {
        Ok(v) => v,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                serde_json::json!({ "error": e.to_string() }).to_string(),
            ).into_response();
        }
    };
    let model_ref = match MODEL.get().cloned() {
        Some(v) => v,
        None => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                serde_json::json!({ "error": "model not init" }).to_string(),
            ).into_response();
        }
    };
    let mut guard = model_ref.write().await;
    let embeddings = match guard.instance.embedding(&texts) {
        Ok(v) => v,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                serde_json::json!({ "error": e.to_string() }).to_string(),
            ).into_response();
        }
    };
    let model_name = guard.which_model.as_string();
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect::<Vec<_>>();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model_name,
    };
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&response).unwrap(),
    ).into_response()
}