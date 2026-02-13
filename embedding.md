# Implementation Plan: Qwen3-Embedding Models (0.6B and 4B)

## Executive Summary

Adding Qwen3-Embedding support to aha by leveraging the existing Qwen3 architecture while creating a new embedding-specific pipeline that returns normalized hidden states instead of logits.

**Key Design Decisions:**
- Reuse existing `Qwen3Config`, `Qwen3DecoderLayer` from `src/models/qwen3/`
- Create new `EmbedModel` trait parallel to `GenerateModel`
- Add OpenAI-compatible `/embeddings` endpoint
- Support MRL (Matryoshka Representation Learning) dimension truncation

---

## Model Specifications

| Property | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B |
|----------|---------------------|-------------------|
| **hidden_size** (embedding dim) | 1024 | 2560 |
| **num_hidden_layers** | 28 | 36 |
| **num_attention_heads** | 16 | 32 |
| **num_key_value_heads** | 8 | 8 |
| **max_position_embeddings** | 32768 | 40960 |
| **vocab_size** | 151669 | 151665 |
| **Architecture** | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` |
| **Pooling** | Last Token (EOS) | Last Token (EOS) |
| **Normalization** | L2 | L2 |

---

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)

| Step | File | Action | Description |
|------|------|--------|-------------|
| 1.1 | `src/models/mod.rs` | Modify | Add `EmbedModel` trait, `EmbeddingParameters`, `EmbeddingResponse`, `EmbeddingInstance` enum |
| 1.2 | `src/models/mod.rs` | Modify | Add `WhichModel::Qwen3Embedding0_6B`, `Qwen3Embedding4B` variants |
| 1.3 | `src/models/mod.rs` | Modify | Add `load_embedding_model()` and `is_embedding_model()` functions |

### Phase 2: Model Implementation (Core Logic)

| Step | File | Action | Description |
|------|------|--------|-------------|
| 2.1 | `src/models/qwen3_embedding/mod.rs` | Create | Module declaration |
| 2.2 | `src/models/qwen3_embedding/config.rs` | Create | `Qwen3EmbeddingConfig` struct |
| 2.3 | `src/models/qwen3_embedding/model.rs` | Create | `Qwen3EmbeddingTransformer`, `Qwen3EmbeddingModel`, `last_token_pool()` |

### Phase 3: API Integration (Server)

| Step | File | Action | Description |
|------|------|--------|-------------|
| 3.1 | `src/api.rs` | Modify | Add `EMBEDDING_MODEL` static, `init_embedding()`, `/embeddings` POST endpoint |
| 3.2 | `src/main.rs` | Modify | Add model IDs, update `run_cli`, `run_serv` for embedding models |

### Phase 4: CLI Support

| Step | File | Action | Description |
|------|------|--------|-------------|
| 4.1 | `src/exec/qwen3_embedding.rs` | Create | CLI exec implementation for `run` subcommand |
| 4.2 | `src/exec/mod.rs` | Modify | Add `pub mod qwen3_embedding` |
| 4.3 | `src/main.rs` | Modify | Add match arm for `run_run` function |

### Phase 5: Testing

| Step | File | Action | Description |
|------|------|--------|-------------|
| 5.1 | `tests/qwen3_embedding_test.rs` | Create | Unit tests for single/batch embedding, MRL, L2 normalization |

---

## Key Technical Details

### Last Token Pooling

```python
# Python (HuggingFace reference)
def last_token_pool(last_hidden_states, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size), sequence_lengths]
```

```rust
// Rust implementation
fn last_token_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (batch_size, _seq_len, _hidden_dim) = hidden_states.dims3()?;
    let sequence_lengths = attention_mask.sum(1)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    
    let mut pooled_outputs = Vec::with_capacity(batch_size);
    for (i, &seq_len) in sequence_lengths.iter().enumerate() {
        let last_idx = (seq_len as usize).saturating_sub(1);
        pooled_outputs.push(hidden_states.i((i, last_idx, ..))?);
    }
    Tensor::stack(&pooled_outputs, 0)
}
```

### Key Differences from Existing Qwen3

| Aspect | Qwen3 (Text) | Qwen3-Embedding |
|--------|--------------|-----------------|
| Output | Logits from `lm_head` | Hidden states from `norm` |
| Pooling | Last token for next-token prediction | Last token pooling for embedding |
| Normalization | Softmax on logits | L2 normalization on embedding |
| lm_head | Required | **Not loaded/used** |

---

## Files Summary

| Action | Count | Files |
|--------|-------|-------|
| **Create** | 5 | `qwen3_embedding/mod.rs`, `config.rs`, `model.rs`, `exec/qwen3_embedding.rs`, `tests/qwen3_embedding_test.rs` |
| **Modify** | 4 | `models/mod.rs`, `api.rs`, `main.rs`, `exec/mod.rs` |

---

## Verification Commands

```bash
# Build
cargo build --features metal  # or cuda

# Test CLI
cargo run --release -- run -m qwen3-embedding-0.6b -i "Hello world"

# Test MRL truncation
cargo run --release -- run -m qwen3-embedding-0.6b -i "Hello" -i "256"

# Test API server
cargo run --release -- serv -m qwen3-embedding-0.6b -p 10100
curl -X POST http://localhost:10100/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding","input":"Hello world"}'
```

---

## OpenAI-Compatible API

### Request Format
```json
{
  "model": "qwen3-embedding-0.6b",
  "input": "Hello world",
  "dimensions": 256
}
```

### Response Format
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "qwen3-embedding",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```
