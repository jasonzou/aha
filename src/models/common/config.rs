//! Common model configuration traits and utilities.
//!
//! Provides shared abstractions for model configurations to reduce duplication
//! and ensure consistency across different model implementations.

use anyhow::Context;
use candle_nn::Activation;
use serde::de::DeserializeOwned;

/// Common configuration properties shared by most transformer-based models.
pub trait ModelConfig: DeserializeOwned + Sized {
    /// Size of the hidden layers.
    fn hidden_size(&self) -> usize;

    /// Number of hidden layers in the transformer.
    fn num_hidden_layers(&self) -> usize;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Maximum position embeddings (context length).
    fn max_position_embeddings(&self) -> usize;

    /// Size of the intermediate layer in the feed-forward network.
    fn intermediate_size(&self) -> usize;

    /// Activation function used in the feed-forward network.
    fn hidden_act(&self) -> Activation;

    /// RMS normalization epsilon.
    fn rms_norm_eps(&self) -> f64;

    /// Whether attention has bias.
    fn attention_bias(&self) -> bool;

    /// Attention dropout probability.
    fn attention_dropout(&self) -> f64;

    /// Initializer range for weight initialization.
    fn initializer_range(&self) -> f64;

    /// Number of key/value heads (for grouped-query attention).
    fn num_key_value_heads(&self) -> usize;

    /// RoPE theta parameter.
    fn rope_theta(&self) -> f32;

    /// Whether word embeddings are tied.
    fn tie_word_embeddings(&self) -> bool;

    /// Torch dtype as string.
    fn torch_dtype(&self) -> &str;

    /// Head dimension size.
    fn head_dim(&self) -> usize;
}

/// Trait for loading configuration from a file.
pub trait FromConfigFile: DeserializeOwned + Sized {
    /// Load configuration from a JSON file.
    fn from_config_file(path: &str) -> anyhow::Result<Self> {
        let config_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;
        let config = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("Failed to parse config file: {}", path))?;
        Ok(config)
    }

    /// Load configuration from a directory (assumes config.json).
    fn from_config_dir(dir_path: &str) -> anyhow::Result<Self> {
        let config_path = format!("{}/config.json", dir_path);
        Self::from_config_file(&config_path)
    }
}

/// Base configuration struct with common fields.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BaseConfig {
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub hidden_size: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub torch_dtype: String,
    pub vocab_size: usize,
}

impl ModelConfig for BaseConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn hidden_act(&self) -> Activation {
        self.hidden_act
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }

    fn attention_bias(&self) -> bool {
        self.attention_bias
    }

    fn attention_dropout(&self) -> f64 {
        self.attention_dropout
    }

    fn initializer_range(&self) -> f64 {
        self.initializer_range
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn torch_dtype(&self) -> &str {
        &self.torch_dtype
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl FromConfigFile for BaseConfig {}

// Default values
fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_initializer_range() -> f64 {
    0.02
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f32 {
    1000000.0
}

fn default_tie_word_embeddings() -> bool {
    true
}

/// Helper macro to implement ModelConfig for existing config structs.
#[macro_export]
macro_rules! impl_model_config {
    ($type:ty) => {
        impl $crate::models::common::config::ModelConfig for $type {
            fn hidden_size(&self) -> usize {
                self.hidden_size
            }

            fn num_hidden_layers(&self) -> usize {
                self.num_hidden_layers
            }

            fn num_attention_heads(&self) -> usize {
                self.num_attention_heads
            }

            fn vocab_size(&self) -> usize {
                self.vocab_size
            }

            fn max_position_embeddings(&self) -> usize {
                self.max_position_embeddings
            }

            fn intermediate_size(&self) -> usize {
                self.intermediate_size
            }

            fn hidden_act(&self) -> candle_nn::Activation {
                self.hidden_act
            }

            fn rms_norm_eps(&self) -> f64 {
                self.rms_norm_eps
            }

            fn attention_bias(&self) -> bool {
                self.attention_bias
            }

            fn attention_dropout(&self) -> f64 {
                self.attention_dropout
            }

            fn initializer_range(&self) -> f64 {
                self.initializer_range
            }

            fn num_key_value_heads(&self) -> usize {
                self.num_key_value_heads
            }

            fn rope_theta(&self) -> f32 {
                self.rope_theta
            }

            fn tie_word_embeddings(&self) -> bool {
                self.tie_word_embeddings
            }

            fn torch_dtype(&self) -> &str {
                &self.torch_dtype
            }

            fn head_dim(&self) -> usize {
                self.head_dim
            }
        }

        impl $crate::models::common::config::FromConfigFile for $type {}
    };
}