//! Shared utilities for model loading and operations.
//!
//! Provides common utilities used across different model implementations
//! to reduce code duplication and ensure consistency.

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Common tensor operations used across models.
pub mod tensor_ops {
    use super::*;

    /// Apply layer normalization with optional bias.
    pub fn layer_norm(
        xs: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> anyhow::Result<Tensor> {
        let bias_tensor = match bias {
            Some(b) => b.clone(),
            None => Tensor::zeros_like(weight)?,
        };
        Ok(candle_nn::ops::layer_norm(
            xs,
            weight,
            &bias_tensor,
            eps as f32,
        )?)
    }

    /// Apply RMS normalization.
    pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> anyhow::Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(xs, weight, eps as f32)?)
    }

    /// Apply rotary positional embeddings.
    pub fn apply_rope(
        query: &Tensor,
        key: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        to_f32: bool,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        use crate::position_embed::rope::apply_rotary_pos_emb;
        apply_rotary_pos_emb(query, key, cos, sin, to_f32)
    }

    /// Create causal attention mask.
    pub fn causal_attention_mask(
        batch_size: usize,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        use crate::utils::tensor_utils::prepare_causal_attention_mask;
        prepare_causal_attention_mask(batch_size, seq_len, 0, device)
            .with_context(|| "Failed to create causal attention mask")
    }
}

/// Common model loading utilities.
pub mod loading {
    use std::collections::HashMap;

    use super::*;

    /// Load configuration from a JSON file with proper error context.
    pub fn load_config<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
        let config_bytes =
            std::fs::read(path).with_context(|| format!("Failed to read config file: {}", path))?;
        let config = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("Failed to parse config file: {}", path))?;
        Ok(config)
    }

    /// Load generation configuration from a JSON file (optional).
    pub fn load_generation_config<T: serde::de::DeserializeOwned>(
        path: &str,
    ) -> anyhow::Result<Option<T>> {
        let config_path = format!("{}/generation_config.json", path);
        if !std::path::Path::new(&config_path).exists() {
            return Ok(None);
        }

        let config_bytes = std::fs::read(&config_path)
            .with_context(|| format!("Failed to read generation config: {}", config_path))?;
        let config = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("Failed to parse generation config: {}", config_path))?;
        Ok(Some(config))
    }

    /// Create a VarBuilder from safetensors files in a directory.
    pub fn vb_from_safetensors_dir<'a>(
        dir_path: &'a str,
        dtype: DType,
        device: &'a Device,
    ) -> anyhow::Result<VarBuilder<'a>> {
        use crate::utils::find_type_files;

        let model_list = find_type_files(dir_path, "safetensors")
            .with_context(|| format!("Failed to find safetensors files in: {}", dir_path))?;

        if model_list.is_empty() {
            return Err(anyhow::anyhow!(
                "No safetensors files found in directory: {}",
                dir_path
            ));
        }

        // Safety: The model files are loaded as memory-mapped and we ensure they exist
        unsafe {
            Ok(VarBuilder::from_mmaped_safetensors(
                &model_list,
                dtype,
                device,
            )?)
        }
    }

    /// Create a VarBuilder from a single model file.
    pub fn vb_from_model_file<'a>(
        model_path: &'a str,
        dtype: DType,
        device: &'a Device,
        key: Option<&'a str>,
    ) -> anyhow::Result<VarBuilder<'a>> {
        let mut dict_to_hashmap = HashMap::new();
        let dict = candle_core::pickle::read_all_with_key(model_path, key)?;
        for (k, v) in dict {
            dict_to_hashmap.insert(k, v);
        }
        Ok(VarBuilder::from_tensors(dict_to_hashmap, dtype, device))
    }
}

/// Common initialization patterns.
pub mod initialization {
    use super::*;
    use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

    /// Initialize tokenizer with error context.
    pub fn init_tokenizer(model_path: &str) -> anyhow::Result<TokenizerModel> {
        TokenizerModel::init(model_path)
            .with_context(|| format!("Failed to initialize tokenizer from: {}", model_path))
    }

    /// Initialize chat template with error context.
    pub fn init_chat_template(model_path: &str) -> anyhow::Result<ChatTemplate<'_>> {
        ChatTemplate::init(model_path)
            .with_context(|| format!("Failed to initialize chat template from: {}", model_path))
    }

    /// Get device with fallback logic.
    pub fn get_device(user_device: Option<&Device>) -> Device {
        match user_device {
            Some(d) => d.clone(),
            None => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).unwrap_or(Device::Cpu)
                }
                #[cfg(all(not(feature = "cuda"), feature = "metal"))]
                {
                    Device::new_metal(0).unwrap_or(Device::Cpu)
                }
                #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
                {
                    Device::Cpu
                }
            }
        }
    }

    /// Get data type with fallback logic.
    pub fn get_dtype(user_dtype: Option<DType>, config_dtype: &str) -> DType {
        match user_dtype {
            Some(d) => d,
            None => {
                #[cfg(feature = "cuda")]
                {
                    match config_dtype {
                        "float32" | "float" => DType::F32,
                        "float64" | "double" => DType::F64,
                        "float16" => DType::F16,
                        "bfloat16" => {
                            // NVIDIA GPUs with SM >= 8.0 support BF16
                            if let Ok(arch) = crate::utils::get_gpu_sm_arch() {
                                if arch >= 8.0 { DType::BF16 } else { DType::F16 }
                            } else {
                                DType::F16
                            }
                        }
                        "uint8" => DType::U8,
                        "int8" | "int16" | "int32" | "int64" => DType::I64,
                        _ => DType::F32,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    match config_dtype {
                        "float32" | "float" => DType::F32,
                        "float64" | "double" => DType::F64,
                        "float16" | "bfloat16" => DType::F16, // BF16 has issues on CPU
                        "uint8" => DType::U8,
                        "int8" | "int16" | "int32" | "int64" => DType::I64,
                        _ => DType::F32,
                    }
                }
            }
        }
    }
}

/// Common error handling for model operations.
pub mod errors {
    use anyhow::Context;

    /// Add context to tensor operations.
    pub fn tensor_op_context<T>(
        result: candle_core::Result<T>,
        operation: &str,
    ) -> anyhow::Result<T> {
        result.with_context(|| format!("Tensor operation failed: {}", operation))
    }

    /// Add context to model loading operations.
    pub fn model_load_context<T>(result: anyhow::Result<T>, model_name: &str) -> anyhow::Result<T> {
        result.with_context(|| format!("Failed to load model: {}", model_name))
    }
}

/// Re-export commonly used items for convenience.
pub use initialization::{get_device, get_dtype, init_chat_template, init_tokenizer};
pub use loading::{load_config, load_generation_config, vb_from_safetensors_dir};
pub use tensor_ops::{apply_rope, causal_attention_mask, layer_norm, rms_norm};
