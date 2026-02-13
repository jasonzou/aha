//! Common model initialization utilities.
//!
//! Provides shared abstractions for model initialization to reduce duplication
//! and ensure consistent initialization patterns across different models.

use candle_core::{DType, Device};
use anyhow::Context;

/// Options for model initialization.
#[derive(Debug, Clone)]
pub struct ModelInitOptions<'a> {
    /// Path to the model directory.
    pub path: &'a str,
    /// Optional device to load the model on.
    pub device: Option<&'a Device>,
    /// Optional data type to use.
    pub dtype: Option<DType>,
    /// Optional key for loading specific tensors.
    pub key: Option<&'a str>,
}

impl<'a> ModelInitOptions<'a> {
    /// Create new initialization options.
    pub fn new(path: &'a str) -> Self {
        Self {
            path,
            device: None,
            dtype: None,
            key: None,
        }
    }

    /// Set the device.
    pub fn with_device(mut self, device: &'a Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set the key.
    pub fn with_key(mut self, key: &'a str) -> Self {
        self.key = Some(key);
        self
    }
}

/// Trait for model initialization with consistent patterns.
pub trait ModelInitializer: Sized {
    /// Initialize the model with the given options.
    fn init(options: ModelInitOptions<'_>) -> anyhow::Result<Self>;

    /// Initialize the model with default options.
    fn init_with_defaults(path: &str) -> anyhow::Result<Self> {
        Self::init(ModelInitOptions::new(path))
    }

    /// Initialize the model with path, device, and dtype (common pattern in existing code).
    fn init_with(path: &str, device: Option<&Device>, dtype: Option<DType>) -> anyhow::Result<Self> {
        let mut options = ModelInitOptions::new(path);
        if let Some(device) = device {
            options = options.with_device(device);
        }
        if let Some(dtype) = dtype {
            options = options.with_dtype(dtype);
        }
        Self::init(options)
    }
}

/// Common utilities for model initialization.
pub mod utils {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    /// Get the device to use, with fallback logic.
    pub fn get_device(device: Option<&Device>) -> Device {
        match device {
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

    /// Get the data type to use, with fallback logic.
    pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
        match dtype {
            Some(d) => d,
            None => {
                #[cfg(feature = "cuda")]
                {
                    match cfg_dtype {
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
                    match cfg_dtype {
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

    /// Find files with a specific extension in a directory.
    pub fn find_type_files(path: &str, extension_type: &str) -> anyhow::Result<Vec<String>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(path)
            .with_context(|| format!("Failed to read directory: {}", path))?
        {
            let entry = entry.with_context(|| "Failed to read directory entry")?;
            let file_path = entry.path();

            if file_path.is_file()
                && let Some(extension) = file_path.extension()
                && extension == extension_type
            {
                files.push(file_path.to_string_lossy().to_string());
            }
        }

        Ok(files)
    }

    /// Create a VarBuilder from safetensors files.
    pub fn create_vb_from_safetensors<'a>(
        path: &'a str,
        dtype: DType,
        device: &'a Device,
    ) -> anyhow::Result<VarBuilder<'a>> {
        let model_list = find_type_files(path, "safetensors")?;
        if model_list.is_empty() {
            return Err(anyhow::anyhow!(
                "No safetensors files found in directory: {}",
                path
            ));
        }

        // Safety: The model files are loaded as memory-mapped and we ensure they exist
        unsafe { Ok(VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)?) }
    }

    /// Create a VarBuilder from a single model file.
    pub fn create_vb_from_model_path<'a>(
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

    /// Create a VarBuilder from files with a specific extension.
    pub fn create_vb_from_extension<'a>(
        path: &'a str,
        extension_type: &'a str,
        dtype: DType,
        device: &'a Device,
        key: Option<&'a str>,
    ) -> anyhow::Result<VarBuilder<'a>> {
        let model_list = find_type_files(path, extension_type)?;
        let mut dict_to_hashmap = HashMap::new();
        for m in model_list {
            let dict = candle_core::pickle::read_all_with_key(&m, key)?;
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        Ok(VarBuilder::from_tensors(dict_to_hashmap, dtype, device))
    }
}

/// Helper macro to implement ModelInitializer for existing model structs.
#[macro_export]
macro_rules! impl_model_initializer {
    ($type:ty, $config_type:ty) => {
        impl $crate::models::common::init::ModelInitializer for $type {
            fn init(options: $crate::models::common::init::ModelInitOptions<'_>) -> anyhow::Result<Self> {
                use $crate::models::common::init::utils;
                use $crate::utils::{get_device, get_dtype};
                use anyhow::Context;

                // Load configuration
                let config_path = format!("{}/config.json", options.path);
                let config: $config_type = <$config_type as $crate::models::common::config::FromConfigFile>::from_config_file(&config_path)
                    .with_context(|| format!("Failed to load config from: {}", config_path))?;

                // Get device and dtype
                let device = utils::get_device(options.device);
                let dtype = utils::get_dtype(options.dtype, config.torch_dtype());

                // Load tokenizer and chat template
                let tokenizer = $crate::tokenizer::TokenizerModel::init(options.path)
                    .with_context(|| "Failed to initialize tokenizer")?;
                let chat_template = $crate::chat_template::ChatTemplate::init(options.path)
                    .with_context(|| "Failed to initialize chat template")?;

                // Create VarBuilder
                let vb = utils::create_vb_from_safetensors(options.path, dtype, &device)
                    .with_context(|| "Failed to create VarBuilder from safetensors")?;

                // Create model
                let model = <$type>::new(&config, vb)
                    .with_context(|| "Failed to create model")?;

                // Load generation config if available
                let generation_config_path = format!("{}/generation_config.json", options.path);
                let generation_config = if std::path::Path::new(&generation_config_path).exists() {
                    let config_bytes = std::fs::read(&generation_config_path)
                        .with_context(|| format!("Failed to read generation config: {}", generation_config_path))?;
                    serde_json::from_slice(&config_bytes)
                        .with_context(|| format!("Failed to parse generation config: {}", generation_config_path))?
                } else {
                    // Default generation config
                    <$type as $crate::models::common::init::ModelInitializerExt>::default_generation_config()
                };

                Ok(<$type>::from_components(
                    chat_template,
                    tokenizer,
                    model,
                    device,
                    generation_config,
                    options.path.to_string(),
                ))
            }
        }
    };
}

/// Extension trait for ModelInitializer with additional utilities.
pub trait ModelInitializerExt: ModelInitializer {
    /// Get the default generation config for this model type.
    fn default_generation_config() -> Self::GenerationConfig
    where
        Self: Sized;

    /// Create model from components (to be implemented by each model).
    fn from_components(
        chat_template: crate::chat_template::ChatTemplate<'_>,
        tokenizer: crate::tokenizer::TokenizerModel,
        model: Self::ModelType,
        device: Device,
        generation_config: Self::GenerationConfig,
        model_name: String,
    ) -> Self;

    /// Associated types
    type ModelType;
    type GenerationConfig: serde::de::DeserializeOwned;
}

/// Common generation configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BaseGenerationConfig {
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
    #[serde(default = "default_do_sample")]
    pub do_sample: bool,
    #[serde(default)]
    pub eos_token_id: Vec<usize>,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

// Default values for generation config
fn default_bos_token_id() -> usize {
    1
}

fn default_pad_token_id() -> usize {
    0
}

fn default_do_sample() -> bool {
    false
}

fn default_top_p() -> f32 {
    1.0
}

fn default_top_k() -> usize {
    50
}

fn default_temperature() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}