use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, conv2d,
    embedding, layer_norm, linear, linear_no_bias, rms_norm,
};

use super::config::{GlmOCRConfig, GlmOCRProjectorConfig, GlmOCRTextConfig, GlmOCRVisionConfig};
use crate::{
    models::common::{GateUpDownMLP, eager_attention_forward},
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
    utils::tensor_utils::prepare_causal_attention_mask,
};

// Vision Encoder Components
pub struct GlmOCRVisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Tensor,
    num_patches: usize,
}

impl GlmOCRVisionEmbeddings {
    pub fn new(vb: VarBuilder, config: &GlmOCRVisionConfig) -> Result<Self> {
        let patch_embedding = conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            Conv2dConfig {
                stride: config.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;

        let num_patches = (config.image_size / config.patch_size).pow(2);
        let position_embedding = vb.get(
            (1, num_patches + 1, config.hidden_size),
            "position_embedding",
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;

        let batch_size = patch_embeds.dim(0)?;
        let cls_embed = self.position_embedding.i((0, 0..1, ..))?;
        let cls_embed = cls_embed.expand((batch_size, 1, cls_embed.dim(1)?))?;
        let pos_embed = self.position_embedding.i((0, 1..=self.num_patches, ..))?;
        let pos_embed = pos_embed.expand((batch_size, self.num_patches, pos_embed.dim(1)?))?;

        let embeddings = Tensor::cat(&[cls_embed, patch_embeds], 1)?;
        let embeddings = embeddings.broadcast_add(&pos_embed)?;
        Ok(embeddings)
    }
}

pub struct GlmOCRVisionAttention {
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    qkv: Linear,
    proj: Linear,
}

impl GlmOCRVisionAttention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            scaling,
            qkv,
            proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv
            .reshape((bs, seq_len, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let attn_output = eager_attention_forward(&q, &k, &v, None, None, self.scaling)?;
        let attn_output = attn_output.reshape((bs, seq_len, ()))?;
        Ok(self.proj.forward(&attn_output)?)
    }
}

pub struct GlmOCRVisionBlock {
    norm1: LayerNorm,
    attn: GlmOCRVisionAttention,
    norm2: LayerNorm,
    mlp: GateUpDownMLP,
}

impl GlmOCRVisionBlock {
    pub fn new(vb: VarBuilder, config: &GlmOCRVisionConfig) -> Result<Self> {
        let norm1 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm1"))?;
        let attn = GlmOCRVisionAttention::new(
            vb.pp("attn"),
            config.hidden_size,
            config.num_attention_heads,
        )?;
        let norm2 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm2"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            true,
            None,
            None,
            None,
        )?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs)?;
        let xs = residual.add(&xs)?;

        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        Ok(xs.add(&residual)?)
    }
}

pub struct GlmOCRVisionEncoder {
    embeddings: GlmOCRVisionEmbeddings,
    blocks: Vec<GlmOCRVisionBlock>,
    post_layernorm: LayerNorm,
}

impl GlmOCRVisionEncoder {
    pub fn new(vb: VarBuilder, config: &GlmOCRVisionConfig) -> Result<Self> {
        let embeddings = GlmOCRVisionEmbeddings::new(vb.pp("embeddings"), config)?;

        let mut blocks = Vec::new();
        for i in 0..config.num_hidden_layers {
            let block = GlmOCRVisionBlock::new(vb.pp("blocks").pp(i), config)?;
            blocks.push(block);
        }

        let post_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_layernorm"),
        )?;

        Ok(Self {
            embeddings,
            blocks,
            post_layernorm,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(pixel_values)?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        Ok(self.post_layernorm.forward(&hidden_states)?)
    }
}

// Projector (Resampler)
pub struct GlmOCRProjector {
    query_embed: Tensor,
    proj: Linear,
    norm: LayerNorm,
    num_queries: usize,
}

impl GlmOCRProjector {
    pub fn new(
        vb: VarBuilder,
        vision_config: &GlmOCRVisionConfig,
        config: &GlmOCRProjectorConfig,
    ) -> Result<Self> {
        let query_embed = vb.get(
            (1, config.num_queries, vision_config.hidden_size),
            "query_embed",
        )?;
        let proj = linear(vision_config.hidden_size, config.hidden_size, vb.pp("proj"))?;
        let norm = layer_norm(config.hidden_size, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            query_embed,
            proj,
            norm,
            num_queries: config.num_queries,
        })
    }

    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let batch_size = image_features.dim(0)?;
        let _queries =
            self.query_embed
                .expand((batch_size, self.num_queries, self.query_embed.dim(2)?))?;

        // Simple projection approach (can be enhanced with cross-attention if needed)
        let projected = self.proj.forward(image_features)?;
        Ok(self.norm.forward(&projected)?)
    }
}

// Language Model Components
pub struct GlmOCRAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
}

impl GlmOCRAttention {
    pub fn new(vb: VarBuilder, config: &GlmOCRTextConfig) -> Result<Self> {
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;
        let head_dim = config.head_dim;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
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
        let (bs, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;

        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;

        let attn_output = attn_output.reshape((bs, q_len, ()))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }
}

pub struct GlmOCRDecoderLayer {
    self_attn: GlmOCRAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl GlmOCRDecoderLayer {
    pub fn new(vb: VarBuilder, config: &GlmOCRTextConfig) -> Result<Self> {
        let self_attn = GlmOCRAttention::new(vb.pp("self_attn"), config)?;
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
        Ok(xs.add(&residual)?)
    }
}

pub struct GlmOCRLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<GlmOCRDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: RoPE,
    config: GlmOCRTextConfig,
}

impl GlmOCRLanguageModel {
    pub fn new(vb: VarBuilder, config: GlmOCRTextConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = GlmOCRDecoderLayer::new(vb.pp("layers").pp(i), &config)?;
            layers.push(layer);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        let rotary_emb = RoPE::new(config.head_dim, config.rope_theta, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        image_features: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // Merge image features if provided
        if let (Some(img_feats), Some(img_mask)) = (image_features, image_mask) {
            let img_mask_bool = img_mask.to_dtype(DType::U8)?.to_vec1::<u8>()?;
            let mut img_idx = 0;
            for (i, &is_image) in img_mask_bool.iter().enumerate() {
                if is_image == 1 && img_idx < img_feats.dim(1)? {
                    let img_feat = img_feats.i((0, img_idx, ..))?;
                    // Use narrow and cat instead of slice_assign
                    let before = inputs_embeds.narrow(1, 0, i)?;
                    let after = inputs_embeds.narrow(1, i + 1, seq_len - i - 1)?;
                    let img_feat_expanded = img_feat.unsqueeze(0)?.unsqueeze(0)?;
                    inputs_embeds = Tensor::cat(&[before, img_feat_expanded, after], 1)?;
                    img_idx += 1;
                }
            }
        }

        let attention_mask = if seq_len > 1 {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                seqlen_offset,
                input_ids.device(),
            )?)
        } else {
            None
        };

        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, input_ids.device())?;

        let mut hidden_states = inputs_embeds;
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }
}

// Full Model
pub struct GlmOCRModel {
    vision_encoder: GlmOCRVisionEncoder,
    projector: GlmOCRProjector,
    language_model: GlmOCRLanguageModel,
}

impl GlmOCRModel {
    pub fn new(vb: VarBuilder, config: GlmOCRConfig) -> Result<Self> {
        let vision_encoder =
            GlmOCRVisionEncoder::new(vb.pp("vision_encoder"), &config.vision_config)?;
        let projector = GlmOCRProjector::new(
            vb.pp("projector"),
            &config.vision_config,
            &config.projector_config,
        )?;
        let language_model = GlmOCRLanguageModel::new(vb.pp("language_model"), config.text_config)?;

        Ok(Self {
            vision_encoder,
            projector,
            language_model,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let image_features = if let Some(pixels) = pixel_values {
            let vision_output = self.vision_encoder.forward(pixels)?;
            let projected = self.projector.forward(&vision_output)?;
            Some(projected)
        } else {
            None
        };

        self.language_model.forward(
            input_ids,
            image_features.as_ref(),
            image_mask,
            seqlen_offset,
        )
    }
}
