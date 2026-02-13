use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder, embedding, rms_norm};

use super::config::Qwen3EmbeddingConfig;
use crate::{
    models::{
        EmbedModel, EmbeddingData, EmbeddingParameters, EmbeddingResponse, EmbeddingUsage,
        common::GateUpDownMLP,
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype, tensor_utils::l2_normalize},
};

pub struct Qwen3EmbeddingAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
}

impl Qwen3EmbeddingAttention {
    pub fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);

        let q_proj = candle_nn::linear_b(
            hidden_size,
            num_attention_heads * head_dim,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            config.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;

        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;

        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;

        let attn_output = crate::models::common::eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;

        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }
}

pub struct Qwen3EmbeddingDecoderLayer {
    self_attn: Qwen3EmbeddingAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3EmbeddingDecoderLayer {
    pub fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3EmbeddingAttention::new(config, vb.pp("self_attn"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
            None,
            None,
            None,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct Qwen3EmbeddingTransformer {
    embed_tokens: Embedding,
    layers: Vec<Qwen3EmbeddingDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: RoPE,
    hidden_size: usize,
}

impl Qwen3EmbeddingTransformer {
    pub fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen3EmbeddingDecoderLayer::new(config, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = RoPE::new(config.head_dim, config.rope_theta, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            hidden_size: config.hidden_size,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;

        let attention_mask: Option<Tensor> = if seq_len > 1 {
            Some(crate::utils::tensor_utils::prepare_causal_attention_mask(
                bs,
                seq_len,
                0,
                input_ids.device(),
            )?)
        } else {
            None
        };

        let (cos, sin) = self.rotary_emb.forward(0, seq_len, input_ids.device())?;

        let mut hidden_states = inputs_embeds;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

fn last_token_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (batch_size, _seq_len, _hidden_dim) = hidden_states.dims3()?;

    let sequence_lengths = attention_mask
        .sum(1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;

    let mut pooled_outputs = Vec::with_capacity(batch_size);
    for (i, &seq_len) in sequence_lengths.iter().enumerate() {
        let last_idx = (seq_len as usize).saturating_sub(1);
        let pooled = hidden_states.i((i, last_idx, ..))?;
        pooled_outputs.push(pooled);
    }

    Ok(Tensor::stack(&pooled_outputs, 0)?)
}

pub struct Qwen3EmbeddingModel {
    transformer: Qwen3EmbeddingTransformer,
    tokenizer: TokenizerModel,
    device: Device,
    model_name: String,
    max_dim: usize,
}

impl Qwen3EmbeddingModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let config_path = format!("{}/config.json", path);
        let config: Qwen3EmbeddingConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;

        let device = get_device(device);
        let cfg_dtype = if config.torch_dtype.is_empty() {
            "bfloat16"
        } else {
            &config.torch_dtype
        };
        let dtype = get_dtype(dtype, cfg_dtype);

        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };

        let transformer = Qwen3EmbeddingTransformer::new(&config, vb)?;
        let tokenizer = TokenizerModel::init(path)?;
        let max_dim = config.embedding_dim();

        Ok(Self {
            transformer,
            tokenizer,
            device,
            model_name: "qwen3-embedding".to_string(),
            max_dim,
        })
    }

    fn create_attention_mask(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(Tensor::ones(
            input_ids.dims(),
            DType::F32,
            input_ids.device(),
        )?)
    }
}

impl EmbedModel for Qwen3EmbeddingModel {
    fn embed(&mut self, params: EmbeddingParameters) -> Result<EmbeddingResponse> {
        let texts = params.input.into_vec();
        let target_dim = params.dimensions.unwrap_or(self.max_dim);

        if target_dim > self.max_dim {
            return Err(anyhow::anyhow!(
                "Requested dimension {} exceeds model's max dimension {}",
                target_dim,
                self.max_dim
            ));
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        let mut total_tokens = 0u32;

        for (idx, text) in texts.iter().enumerate() {
            let input_ids = self.tokenizer.text_encode(text.clone(), &self.device)?;
            let seq_len = input_ids.dim(1)? as u32;
            total_tokens += seq_len;

            let attention_mask = self.create_attention_mask(&input_ids)?;
            let hidden_states = self.transformer.forward(&input_ids)?;
            let pooled = last_token_pool(&hidden_states, &attention_mask)?;
            let normalized = l2_normalize(&pooled, 1)?;

            let truncated = if target_dim < self.max_dim {
                let t = normalized.narrow(1, 0, target_dim)?;
                l2_normalize(&t, 1)?
            } else {
                normalized
            };

            let embedding_vec = truncated
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;

            embeddings.push(EmbeddingData {
                object: "embedding".to_string(),
                embedding: embedding_vec,
                index: idx,
            });
        }

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data: embeddings,
            model: self.model_name.clone(),
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        })
    }
}
