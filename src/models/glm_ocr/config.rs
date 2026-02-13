use candle_nn::Activation;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
}

fn default_hidden_act() -> Activation {
    Activation::Gelu
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRProjectorConfig {
    pub hidden_size: usize,
    pub projector_hidden_act: Activation,
    pub num_queries: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
}

fn default_text_hidden_act() -> Activation {
    Activation::Silu
}

fn default_use_cache() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRConfig {
    pub vision_config: GlmOCRVisionConfig,
    pub projector_config: GlmOCRProjectorConfig,
    pub text_config: GlmOCRTextConfig,
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    #[serde(default)]
    pub torch_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRGenerationConfig {
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    #[serde(default)]
    pub do_sample: bool,
    pub eos_token_id: Vec<usize>,
    pub top_p: f32,
    pub top_k: usize,
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_repetition_penalty() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOCRPreprocessorConfig {
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
}

fn default_image_size() -> usize {
    448
}
