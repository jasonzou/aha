use serde::Serialize;

#[derive(Debug, serde::Deserialize)]
pub struct TranscriptionRequest {
    pub file: Option<String>,
    pub model: Option<String>,
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TranscriptionResponse {
    pub(crate) text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub(crate) error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorDetail {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) code: Option<String>,
}