use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use aha::models::{GenerateModel, ModelInstance, common::model_mapping::WhichModel, load_model};
use aha::params::chat::ChatCompletionParameters;
use aha::utils::string_to_static_str;
use anyhow::anyhow;
use axum::{
    extract::Json as AxumJson,
    response::IntoResponse,
};
use tokio::sync::RwLock;
use serde::Serialize;

use crate::server::process::cleanup_pid_file;

pub(crate) struct StoredModel {
    pub which_model: WhichModel,
    pub instance: ModelInstance<'static>,
}

pub(crate) static MODEL: OnceLock<Arc<RwLock<StoredModel>>> = OnceLock::new();
static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();
static ALLOW_REMOTE_SHUTDOWN: OnceLock<bool> = OnceLock::new();

pub fn init(
    model_type: WhichModel,
    path: String,
    _gguf: Option<String>,
    _mmproj: Option<String>,
) -> anyhow::Result<()> {
    if model_type.is_gguf() {
        return Err(anyhow!("gguf models not supported in this build"));
    }
    if model_type.is_onnx() {
        return Err(anyhow!("onnx not supported"));
    }
    let model_path = string_to_static_str(path);
    let model = load_model(model_type, model_path, None, None)?;

    MODEL.get_or_init(|| {
        Arc::new(RwLock::new(StoredModel {
            which_model: model_type,
            instance: model,
        }))
    });
    Ok(())
}

pub fn set_server_port(port: u16, allow_remote_shutdown: bool) {
    SHUTDOWN_FLAG.get_or_init(|| Arc::new(AtomicBool::new(false)));
    SERVER_PORT.get_or_init(|| port);
    ALLOW_REMOTE_SHUTDOWN.get_or_init(|| allow_remote_shutdown);
}

#[allow(unused)]
pub fn get_shutdown_flag() -> Arc<AtomicBool> {
    SHUTDOWN_FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

#[derive(Serialize)]
pub(crate) struct ErrorResponse {
    error: String,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        let json = serde_json::to_string(&self).unwrap();
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            json,
        ).into_response()
    }
}

#[axum::debug_handler]
pub(crate) async fn chat(
    AxumJson(req): AxumJson<ChatCompletionParameters>,
) -> impl IntoResponse {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        let mut guard = model_ref.write().await;
        guard.instance.generate(req)
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (
                axum::http::StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                response_str,
            ).into_response()
        }
        Err(e) => ErrorResponse { error: e.to_string() }.into_response(),
    }
}

pub(crate) async fn remove_background(
    AxumJson(req): AxumJson<ChatCompletionParameters>,
) -> impl IntoResponse {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        let mut guard = model_ref.write().await;
        guard.instance.generate(req)
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (
                axum::http::StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                response_str,
            ).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            format!("{{\"error\": \"{}\"}}", e),
        ).into_response(),
    }
}

pub(crate) async fn speech(
    AxumJson(req): AxumJson<ChatCompletionParameters>,
) -> impl IntoResponse {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        let mut guard = model_ref.write().await;
        guard.instance.generate(req)
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (
                axum::http::StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                response_str,
            ).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            format!("{{\"error\": \"{}\"}}", e),
        ).into_response(),
    }
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub(crate) async fn health() -> impl IntoResponse {
    if MODEL.get().is_some() {
        let response = HealthResponse {
            status: "ok".to_string(),
            error: None,
        };
        (axum::http::StatusCode::OK, AxumJson(response)).into_response()
    } else {
        let response = HealthResponse {
            status: "unhealthy".to_string(),
            error: Some("model not initialized".to_string()),
        };
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, AxumJson(response)).into_response()
    }
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: Option<i64>,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ErrorResp {
    error: String,
}

pub(crate) async fn models() -> impl IntoResponse {
    if let Some(model_ref) = MODEL.get() {
        let guard = model_ref.read().await;
        let which_model = guard.which_model;

        let model_obj = ModelObject {
            id: which_model.as_string(),
            object: "model".to_string(),
            created: None,
            owned_by: which_model.model_owner(),
        };
        drop(guard);

        let response = ModelsListResponse {
            object: "list".to_string(),
            data: vec![model_obj],
        };
        (axum::http::StatusCode::OK, AxumJson(response)).into_response()
    } else {
        let response = ErrorResp {
            error: "model not initialized".to_string(),
        };
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, AxumJson(response)).into_response()
    }
}

#[derive(Serialize)]
struct ShutdownResponse {
    message: String,
}

pub(crate) async fn shutdown() -> impl IntoResponse {
    let allow_remote = ALLOW_REMOTE_SHUTDOWN.get().copied().unwrap_or(false);

    eprintln!(
        "[SHUTDOWN] Shutdown requested (remote_allowed: {})",
        allow_remote
    );

    if let Some(flag) = SHUTDOWN_FLAG.get() {
        flag.store(true, Ordering::SeqCst);
    }

    if let Some(&port) = SERVER_PORT.get() {
        let _ = cleanup_pid_file(port);
    }

    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        std::process::exit(0);
    });

    let response = ShutdownResponse {
        message: "Shutting down...".to_string(),
    };
    (axum::http::StatusCode::OK, AxumJson(response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint_uninitialized() {
        let response = health().await.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_models_endpoint_uninitialized() {
        let response = models().await.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_get_model_type_asr() {
        assert_eq!(WhichModel::Qwen3ASR0_6B.model_type(), "asr");
        assert_eq!(WhichModel::Qwen3ASR1_7B.model_type(), "asr");
        assert_eq!(WhichModel::GlmASRNano2512.model_type(), "asr");
        assert_eq!(WhichModel::FunASRNano2512.model_type(), "asr");
    }
}