use serde::Deserialize;

use crate::models::gpt2::config::GPT2Config;

#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerConfig {
    pub sample_rate: usize,
    pub sampling_rate: usize,
    pub downsample_rate: usize,
    pub causal_transformer_context_duration: f64,
    pub number_channels: usize,
    pub enable_channel_interleave: bool,
    pub compute_dtype: String,
    pub dtype: String,
    pub code_dim: usize,
    pub encoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
    pub decoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
    pub quantizer_type: String,
    pub quantizer_kwargs: MossAudioTokenizerQuantizerKwargs,
    pub reversed_decoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
}


#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerModuleConfig {
    pub module_type: String,
    pub patch_size: Option<usize>,
    pub causal: Option<bool>,
    pub context_duration: Option<f64>,
    pub conv_layout: Option<bool>,
    pub d_model: Option<usize>,
    pub dim_feedforward: Option<usize>,
    pub gating: Option<String>,
    pub input_dimension: Option<usize>,
    pub layer_scale: Option<f64>,
    pub max_period: Option<usize>,
    pub norm: Option<String>,
    pub num_heads: Option<usize>,
    pub num_layers: Option<usize>,
    pub output_dimension: Option<usize>,
    pub positional_embedding: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerQuantizerKwargs {
    pub codebook_dim: usize,
    pub codebook_loss_weight: f64,
    pub codebook_size: usize,
    pub commitment_loss_weight: f64,
    pub input_dim: usize,
    pub num_quantizers: usize,
    pub output_dim: usize,
    pub quantizer_dropout: f64,
    pub quantizer_type: String,
    pub rvq_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct MossTTSConfig {
    pub add_cross_attention: bool,
    // Audio Tokenizer Specifics
    pub audio_assistant_slot_token_id: u32,
    pub audio_codebook_sizes: Vec<usize>,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_tokenizer_sample_rate: usize,
    pub audio_user_slot_token_id: u32,
    pub audio_vocab_size: usize,
    
    // Generation/Model Params (Simplified nullables to Options or defaults if not critical)
    pub bad_words_ids: Option<Vec<u32>>,
    pub begin_suppress_tokens: Option<Vec<u32>>,
    pub bos_token_id: Option<u32>,
    pub chunk_size_feed_forward: usize,
    pub cross_attention_hidden_size: Option<usize>,
    pub decoder_start_token_id: Option<u32>,
    pub diversity_penalty: f64,
    pub do_sample: bool,
    pub dtype: String,
    pub early_stopping: bool,
    pub encoder_no_repeat_ngram_size: usize,
    pub eos_token_id: Option<u32>,
    pub exponential_decay_length_penalty: Option<f64>,
    pub finetuning_task: Option<String>,
    pub forced_bos_token_id: Option<u32>,
    pub forced_eos_token_id: Option<u32>,
    
    // GPT2 Backbone Config
    pub gpt2_config: GPT2Config,
    
    pub hidden_size: usize,
    pub id2label: std::collections::HashMap<usize, String>,
    
    pub im_end_token_id: u32,
    pub im_start_token_id: u32,
    pub initializer_range: f64,
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    pub label2id: std::collections::HashMap<String, usize>,
    
    pub length_penalty: f64,
    pub local_transformer_attn_implementation: String,
    pub local_transformer_layers: usize,
    
    pub max_length: usize,
    pub max_position_embeddings: usize,
    pub min_length: usize,
    
    pub model_architecture: String,
    pub model_type: String,
    
    pub n_vq: usize,
    pub no_repeat_ngram_size: usize,
    
    pub num_beam_groups: usize,
    pub num_beams: usize,
    pub num_return_sequences: usize,
    
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub output_scores: bool,
    
    pub pad_token_id: u32,
    pub prefix: Option<String>,
    pub problem_type: Option<String>,
    // pub pruned_heads: std::collections::HashMap<String, Vec<usize>>,
    
    pub remove_invalid_values: bool,
    pub repetition_penalty: f64,
    
    pub return_dict: bool,
    pub return_dict_in_generate: bool,
    
    pub sep_token_id: Option<u32>,
    pub suppress_tokens: Option<Vec<u32>>,
    
    pub task_specific_params: Option<serde_json::Value>,
    
    pub temperature: f32,
    pub tf_legacy_loss: bool,
    pub tie_encoder_decoder: bool,
    pub tie_word_embeddings: bool,
    
    pub tokenizer_class: String,
    pub tokenizer_use_fast: bool,
    
    pub top_k: usize,
    pub top_p: f32,
    pub torchscript: bool,
    
    pub typical_p: f64,
    
    pub use_bfloat16: bool,
    pub vocab_size: usize,
    
}


