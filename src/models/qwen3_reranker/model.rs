use anyhow::Result;
use candle_core::{DType, Device, IndexOp};
use candle_nn::{Module, VarBuilder, linear_no_bias};

use super::config::Qwen3RerankerConfig;
use crate::{
    models::{
        RerankModel, RerankParameters, RerankResponse, RerankResult,
        qwen3_embedding::model::Qwen3EmbeddingTransformer,
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Qwen3RerankerModel {
    transformer: Qwen3EmbeddingTransformer,
    classifier: candle_nn::Linear,
    tokenizer: TokenizerModel,
    device: Device,
    model_name: String,
}

impl Qwen3RerankerModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let config_path = format!("{}/config.json", path);
        let config: Qwen3RerankerConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;

        let device = get_device(device);
        let cfg_dtype = if config.torch_dtype.is_empty() {
            "bfloat16"
        } else {
            &config.torch_dtype
        };
        let dtype = get_dtype(dtype, cfg_dtype);

        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };

        let transformer = Qwen3EmbeddingTransformer::new(
            &crate::models::qwen3_embedding::config::Qwen3EmbeddingConfig {
                attention_bias: config.attention_bias,
                attention_dropout: config.attention_dropout,
                head_dim: config.head_dim,
                hidden_act: config.hidden_act,
                hidden_size: config.hidden_size,
                initializer_range: config.initializer_range,
                intermediate_size: config.intermediate_size,
                max_position_embeddings: config.max_position_embeddings,
                num_attention_heads: config.num_attention_heads,
                num_hidden_layers: config.num_hidden_layers,
                num_key_value_heads: config.num_key_value_heads,
                rms_norm_eps: config.rms_norm_eps,
                rope_theta: config.rope_theta,
                tie_word_embeddings: config.tie_word_embeddings,
                torch_dtype: config.torch_dtype.clone(),
                vocab_size: config.vocab_size,
            },
            vb.pp("model"),
        )?;

        // Classification head: hidden_size -> num_labels (typically 1)
        let classifier =
            linear_no_bias(config.hidden_size, config.num_labels, vb.pp("classifier"))?;

        let tokenizer = TokenizerModel::init(path)?;

        Ok(Self {
            transformer,
            classifier,
            tokenizer,
            device,
            model_name: "qwen3-reranker".to_string(),
        })
    }

    /// Rerank documents for a given query
    /// Returns relevance scores for each document (higher = more relevant)
    pub fn compute_scores(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(documents.len());

        for doc in documents {
            // Format: query [SEP] document (or use tokenizer's template if available)
            let input_text = format!("{}</s>{}", query, doc);

            let input_ids = self.tokenizer.text_encode(input_text, &self.device)?;
            let hidden_states = self.transformer.forward(&input_ids)?;

            // Use last token representation (similar to embedding model)
            // Shape: (batch=1, seq_len, hidden_size)
            let (_, seq_len, _) = hidden_states.dims3()?;
            let last_token = hidden_states.i((0, seq_len - 1, ..))?; // Last token

            // Pass through classifier
            let logits = self.classifier.forward(&last_token)?;

            // Apply sigmoid to get probability score
            let score = candle_nn::ops::sigmoid(&logits)?;
            let score_val = score.to_dtype(DType::F32)?.to_vec1::<f32>()?[0];

            scores.push(score_val);
        }

        Ok(scores)
    }

    /// Rerank with batch processing for efficiency
    pub fn compute_scores_batch(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        // For simplicity, process one at a time for now
        // Can be optimized to batch process if needed
        self.compute_scores(query, documents)
    }
}

impl RerankModel for Qwen3RerankerModel {
    fn rerank(&mut self, params: RerankParameters) -> Result<RerankResponse> {
        let scores = self.compute_scores(&params.query, &params.documents)?;

        // Create results with indices
        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| RerankResult {
                index: idx,
                relevance_score: score,
                document: if params.return_documents {
                    Some(params.documents[idx].clone())
                } else {
                    None
                },
            })
            .collect();

        // Sort by relevance score (descending)
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top_k limit
        if params.top_k < results.len() {
            results.truncate(params.top_k);
        }

        Ok(RerankResponse {
            model: self.model_name.clone(),
            results,
        })
    }
}
