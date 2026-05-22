use axum::Json as AxumJson;
use axum::response::IntoResponse;
use serde::Deserialize;

use crate::{
    params::rerank::{RerankResponse, RerankResult},
    server::api::MODEL,
};

fn validate_rerank_input(query: &str, documents: &[String]) -> anyhow::Result<()> {
    if query.trim().is_empty() {
        return Err(anyhow::anyhow!("rerank query cannot be empty"));
    }
    if documents.is_empty() {
        return Err(anyhow::anyhow!("rerank documents cannot be empty"));
    }
    if documents.iter().any(|doc| doc.trim().is_empty()) {
        return Err(anyhow::anyhow!(
            "rerank documents cannot contain empty strings"
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
pub(crate) struct RerankRequestJson {
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
}

pub(crate) async fn rerank(
    AxumJson(req): AxumJson<RerankRequestJson>,
) -> axum::response::Response {
    if let Err(e) = validate_rerank_input(&req.query, &req.documents) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "error": e.to_string() }).to_string(),
        ).into_response();
    }

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
    let scores = match guard.instance.rerank(&req.query, &req.documents) {
        Ok(v) => v,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                serde_json::json!({ "error": e.to_string() }).to_string(),
            ).into_response();
        }
    };

    let mut results = scores
        .into_iter()
        .enumerate()
        .map(|(index, relevance_score)| RerankResult {
            index,
            relevance_score,
            document: req.documents[index].clone(),
        })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = req.top_n {
        results.truncate(top_n.min(results.len()));
    }

    let response = RerankResponse {
        object: "list".to_string(),
        model: guard.which_model.as_string(),
        results,
    };
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&response).unwrap(),
    ).into_response()
}