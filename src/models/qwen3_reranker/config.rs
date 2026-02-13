use candle_nn::Activation;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3RerankerConfig {
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
    #[serde(default = "default_num_labels")]
    pub num_labels: usize,
}

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

fn default_num_labels() -> usize {
    1
}
