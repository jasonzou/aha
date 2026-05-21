//! CLI exec module for direct model inference
//!
//! This module provides model-specific exec implementations for the `run` subcommand.
//! Each model has its own exec module that handles input/output parsing and model invocation.


pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;

use anyhow::Result;

/// Trait for model exec implementations
///
/// Each model exec module implements this trait to provide
/// model-specific inference logic for CLI `run` commands.
pub trait ExecModel {
    /// Run inference with the given input and output parameters
    ///
    /// # Arguments
    /// * `input` - Input text or file path (interpretation is model-specific)
    /// * `output` - Optional output file path (if None, model will auto-generate)
    /// * `weight_path` - Path to the model weights
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(anyhow::Error)` on failure
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()>;
}
