# Qwen3-Rerankers Implementation Plan

## Overview

Qwen3-Rerankers are cross-encoder models that score the relevance between queries and documents. Unlike embedding models (bi-encoders), rerankers process query-document pairs together to produce a relevance score. This is useful for RAG (Retrieval-Augmented Generation) pipelines to improve retrieval quality.

## Model Information

Based on Qwen3 architecture, reranker models typically include:
- Base transformer (similar to Qwen3 embedding model)
- A classification head on top of the [CLS] token or pooled representation
- Output: single logit or probability score for relevance

Expected ModelScope IDs:
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B` (if available)

## Implementation Plan

### Phase 1: Core Model Implementation

#### 1.1 Create Model Module Structure
```
src/models/qwen3_reranker/
├── mod.rs       # Module exports
├── config.rs    # Model configuration
├── model.rs     # Transformer + classification head
└── generate.rs  # RerankModel trait implementation
```

#### 1.2 Configuration (`config.rs`)
```rust
pub struct Qwen3RerankerConfig {
    // Same as Qwen3EmbeddingConfig
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub vocab_size: usize,
    // Reranker-specific
    pub num_labels: usize,  // Typically 1 for relevance scoring
}
```

#### 1.3 Model Architecture (`model.rs`)

Reuse `Qwen3EmbeddingTransformer` from `qwen3_embedding`:
- Same transformer architecture (embeddings, layers, norm)
- Same RoPE positional embeddings
- Same attention mechanism

Add classification head:
```rust
pub struct Qwen3RerankerModel {
    transformer: Qwen3EmbeddingTransformer,
    classifier: candle_nn::Linear,  // hidden_size -> 1
    tokenizer: TokenizerModel,
    device: Device,
    model_name: String,
}
```

Forward pass:
1. Tokenize query+document pairs with separator token
2. Pass through transformer
3. Pool using [CLS] token or mean pooling
4. Pass through classifier to get relevance score
5. Apply sigmoid to get probability (0-1)

#### 1.4 RerankModel Trait (`src/models/mod.rs`)

Add new trait and types:
```rust
// Reranker input - query + list of documents
#[derive(Debug, Clone, Deserialize)]
pub struct RerankParameters {
    pub query: String,
    pub documents: Vec<String>,
    pub model: String,
    pub top_k: Option<usize>,
    pub return_documents: Option<bool>,
}

// Rerank result for a single document
#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
}

pub trait RerankModel {
    fn rerank(&mut self, params: RerankParameters) -> Result<RerankResponse>;
}
```

### Phase 2: Integration with Model System

#### 2.1 Update `src/models/mod.rs`

1. Add module declaration:
```rust
pub mod qwen3_reranker;
```

2. Add WhichModel variants:
```rust
#[value(name = "qwen3-reranker-0.6b")]
Qwen3Reranker0_6B,
#[value(name = "qwen3-reranker-4b")]
Qwen3Reranker4B,
```

3. Add `is_reranker_model()` function:
```rust
pub fn is_reranker_model(model_type: WhichModel) -> bool {
    matches!(
        model_type,
        WhichModel::Qwen3Reranker0_6B | WhichModel::Qwen3Reranker4B
    )
}
```

4. Add to `ModelInstance` enum (or create separate handling like embedding models)

Note: Rerankers should be handled separately like embedding models since they have a different interface.

### Phase 3: API Layer

#### 3.1 Update `src/api.rs`

1. Add static model storage:
```rust
static RERANK_MODEL: OnceLock<Arc<RwLock<Qwen3RerankerModel>>> = OnceLock::new();
```

2. Add init function:
```rust
pub fn init_rerank(path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = Qwen3RerankerModel::init(model_path, None, None)?;
    RERANK_MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}
```

3. Add rerank endpoint:
```rust
#[post("/rerank", data = "<req>")]
pub(crate) async fn rerank(req: Json<RerankParameters>) -> (Status, String) {
    let response = {
        let model_ref = RERANK_MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("rerank model not init"))
            .unwrap();
        model_ref.write().await.rerank(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}
```

### Phase 4: CLI Integration

#### 4.1 Create Exec Module (`src/exec/qwen3_reranker.rs`)

```rust
pub struct Qwen3RerankerExec;

impl ExecModel for Qwen3RerankerExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        // Parse input: query and documents (file or inline)
        // Run reranking
        // Display or save results
    }
}
```

#### 4.2 Update `src/exec/mod.rs`
```rust
pub mod qwen3_reranker;
```

#### 4.3 Update `src/main.rs`

1. Add model IDs:
```rust
WhichModel::Qwen3Reranker0_6B => "Qwen/Qwen3-Reranker-0.6B",
WhichModel::Qwen3Reranker4B => "Qwen/Qwen3-Reranker-4B",
```

2. Add to `run_list()`

3. Add to `run_run()` match arm:
```rust
WhichModel::Qwen3Reranker0_6B | WhichModel::Qwen3Reranker4B => {
    use aha::exec::qwen3_reranker::Qwen3RerankerExec;
    Qwen3RerankerExec::run(&input, output.as_deref(), &weight_path)?;
}
```

4. Update `run_cli()` and `run_serv()` to handle reranker models:
```rust
let is_reranker = models::is_reranker_model(common.model);
if is_embedding {
    init_embedding(model_path)?;
} else if is_reranker {
    init_rerank(model_path)?;
} else {
    init(common.model, model_path)?;
}
```

5. Update `start_http_server()` to mount rerank endpoint:
```rust
if is_embedding {
    builder = builder.mount("/", routes![api::embeddings]);
} else if is_reranker {
    builder = builder.mount("/", routes![api::rerank]);
} else {
    // ... existing routes
}
```

### Phase 5: Testing

#### 5.1 Create Integration Test (`tests/test_qwen3_reranker.rs`)

```rust
#[test]
fn qwen3_reranker_0_6b_load() {
    // Test model loading
}

#[test]
fn qwen3_reranker_0_6b_rerank() {
    // Test basic reranking with query and documents
}
```

## Key Design Decisions

1. **Reuse Transformer**: Use `Qwen3EmbeddingTransformer` from `qwen3_embedding` to avoid code duplication.

2. **Separate Model Instance**: Handle rerankers like embedding models (separate static storage) rather than `GenerateModel` trait since the interface is fundamentally different.

3. **Input Format**: Support both:
   - Direct text pairs (CLI)
   - JSON API format (HTTP)

4. **Tokenization**: Use the same tokenizer as Qwen3, with special handling for query-document separator tokens.

5. **Batch Processing**: Process documents in batches for efficiency.

## API Usage Examples

### HTTP API
```bash
POST /rerank
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of AI...",
    "The weather today is sunny...",
    "Deep learning uses neural networks..."
  ],
  "model": "qwen3-reranker-0.6b"
}

Response:
{
  "model": "qwen3-reranker-0.6b",
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.87},
    {"index": 1, "relevance_score": 0.12}
  ]
}
```

### CLI
```bash
# Rerank documents
aha run -m qwen3-reranker-0.6b -i "query: What is ML?" -i "doc1: Machine learning..." -i "doc2: Weather today..."

# Or from file
aha run -m qwen3-reranker-0.6b -i "file://query.txt" -i "file://docs.json"
```

## Implementation Checklist

- [ ] Create `src/models/qwen3_reranker/` module
- [ ] Implement `config.rs` with reranker-specific config
- [ ] Implement `model.rs` with transformer + classifier
- [ ] Add `RerankModel` trait and types to `src/models/mod.rs`
- [ ] Add `WhichModel` variants
- [ ] Add `is_reranker_model()` function
- [ ] Update `src/api.rs` with rerank endpoint
- [ ] Create `src/exec/qwen3_reranker.rs`
- [ ] Update `src/exec/mod.rs`
- [ ] Update `src/main.rs` with model IDs and routing
- [ ] Create integration tests
- [ ] Update documentation

## Notes

- The actual ModelScope IDs should be verified once models are released
- Tokenizer special tokens (like `[CLS]`, `[SEP]`) need to be checked in the actual model config
- The classification head architecture (single linear vs MLP) should match the original implementation
