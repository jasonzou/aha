//! Error handling utilities for consistent error management across the codebase.
//!
//! Provides extensions to `Result` for better error context and standardized
//! API error responses.

use rocket::http::Status;
use serde::Serialize;

// Re-export anyhow::Context for convenience
pub use anyhow::Context;

/// Creates a standardized API error response.
///
/// # Arguments
/// * `error` - The error message or object to convert to string
///
/// # Returns
/// A tuple of `(Status, String)` suitable for Rocket responses.
pub fn api_error<E: std::fmt::Display>(error: E) -> (Status, String) {
    (Status::InternalServerError, error.to_string())
}

/// Creates a standardized API JSON response.
///
/// # Arguments
/// * `result` - Serializable result to convert to JSON
///
/// # Returns
/// A tuple of `(Status, String)` with the JSON response or error.
pub fn api_json_result<T: Serialize>(result: T) -> (Status, String) {
    match serde_json::to_string(&result) {
        Ok(json) => (Status::Ok, json),
        Err(e) => api_error(e),
    }
}

/// Standardized API error structure for consistent error responses.
#[derive(Debug, Serialize)]
pub struct ApiError {
    /// Human-readable error message
    pub error: String,
    /// HTTP status code
    pub code: u16,
}

impl ApiError {
    /// Creates a new API error with custom message and code.
    pub fn new(error: impl Into<String>, code: u16) -> Self {
        Self {
            error: error.into(),
            code,
        }
    }

    /// Creates a 500 Internal Server Error.
    pub fn internal(error: impl Into<String>) -> Self {
        Self::new(error, 500)
    }

    /// Creates a 404 Not Found error.
    pub fn not_found(error: impl Into<String>) -> Self {
        Self::new(error, 404)
    }

    /// Creates a 400 Bad Request error.
    pub fn bad_request(error: impl Into<String>) -> Self {
        Self::new(error, 400)
    }

    /// Converts the API error to a Rocket response tuple.
    pub fn to_response(&self) -> (Status, String) {
        let status = match self.code {
            400 => Status::BadRequest,
            404 => Status::NotFound,
            500 => Status::InternalServerError,
            _ => Status::InternalServerError,
        };

        match serde_json::to_string(self) {
            Ok(json) => (status, json),
            Err(e) => (Status::InternalServerError, e.to_string()),
        }
    }
}

/// Helper function to create an internal server error response.
pub fn internal_error<E: std::fmt::Display>(error: E) -> (Status, String) {
    ApiError::internal(error.to_string()).to_response()
}

/// Helper function to create a not found error response.
pub fn not_found_error<E: std::fmt::Display>(error: E) -> (Status, String) {
    ApiError::not_found(error.to_string()).to_response()
}

/// Helper function to create a bad request error response.
pub fn bad_request_error<E: std::fmt::Display>(error: E) -> (Status, String) {
    ApiError::bad_request(error.to_string()).to_response()
}