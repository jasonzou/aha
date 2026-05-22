use axum::body::Body;
use axum::response::IntoResponse;

use crate::params::asr::{ErrorDetail, ErrorResponse, TranscriptionResponse};
use crate::server::api::MODEL;

fn error_response(
    status: axum::http::StatusCode,
    error_type: &str,
    message: &str,
    code: Option<String>,
) -> axum::response::Response {
    let error_response = ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            error_type: error_type.to_string(),
            code,
        },
    };
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&error_response).unwrap(),
    ).into_response()
}

pub(crate) async fn transcriptions(
    body: Body,
) -> axum::response::Response {
    use http_body_util::BodyExt;
    
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes().to_vec(),
        Err(e) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Failed to read body: {}", e),
                Some("body_read_error".to_string()),
            );
        }
    };

    let mut file_bytes: Option<Vec<u8>> = None;

    if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
        let header_end = pos + 4;
        file_bytes = Some(bytes[header_end..].to_vec());
    }

    let file_bytes = match file_bytes {
        Some(bytes) => bytes,
        None => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Audio file is required",
                Some("missing_file".to_string()),
            );
        }
    };

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!("aha_upload_{}", uuid::Uuid::new_v4()));
    
    if let Err(e) = tokio::fs::write(&file_path, &file_bytes).await {
        return error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &format!("Failed to save audio file: {}", e),
            Some("file_save_error".to_string()),
        );
    }

    let file_url = format!("file://{}", file_path.display());

    use aha::models::GenerateModel;
    use aha::params::chat::{
        AudioUrlType, ChatCompletionParameters, ChatMessage, ChatMessageAudioContentPart,
        ChatMessageContent, ChatMessageContentPart,
    };

    let audio_part = ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
        r#type: "audio".to_string(),
        audio_url: AudioUrlType { url: file_url },
    });

    let mut params = ChatCompletionParameters::default();
    params.messages = vec![ChatMessage::User {
        content: ChatMessageContent::ContentPart(vec![audio_part]),
        name: None,
    }];
    params.model = "asr".to_string();
    params.temperature = Some(0.0);

    let model_ref = match MODEL.get() {
        Some(m) => m,
        None => {
            let _ = tokio::fs::remove_file(&file_path).await;
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                "Model not initialized",
                Some("model_not_loaded".to_string()),
            );
        }
    };

    let response = {
        let mut guard = model_ref.write().await;
        guard.instance.generate(params)
    };

    let _ = tokio::fs::remove_file(&file_path).await;

    match response {
        Ok(chat_response) => {
            let raw_text = chat_response
                .choices
                .first()
                .and_then(|choice| {
                    if let ChatMessage::Assistant { content, .. } = &choice.message {
                        content.as_ref().and_then(|c| {
                            if let ChatMessageContent::Text(text) = c {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_else(String::new);

            use aha::utils::clean_asr_response;
            let cleaned_text = clean_asr_response(&raw_text);

            let transcription = TranscriptionResponse { text: cleaned_text };
            let json = serde_json::to_string(&transcription).unwrap();
            (
                axum::http::StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json,
            ).into_response()
        }
        Err(e) => {
            let error_msg = e.to_string();
            let (status, error_type, code) =
                if error_msg.contains("audio") || error_msg.contains("decode") {
                    (
                        axum::http::StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        Some("audio_decode_error".to_string()),
                    )
                } else {
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        Some("inference_error".to_string()),
                    )
                };

            error_response(status, error_type, &error_msg, code)
        }
    }
}