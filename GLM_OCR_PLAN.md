# GLM-OCR Implementation Plan

## Overview

GLM-OCR is a vision-language model by ZhipuAI for OCR (Optical Character Recognition) tasks. Based on the patterns from existing OCR models (DeepSeek-OCR, Hunyuan-OCR), GLM-OCR likely combines:
- Vision encoder (ViT-based image encoder)
- Projector to align vision and language features
- Language model (GLM-based text decoder)

## Model Information

Expected ModelScope ID: `ZhipuAI/GLM-OCR` (or similar)

Architecture (estimated based on GLM family):
- Vision Encoder: ViT (Vision Transformer) for image patch encoding
- Perceiver/Projector: Resampler or MLP to compress image tokens
- Language Model: GLM-style transformer with multi-query attention

## Implementation Plan

### Phase 1: Core Model Implementation

#### 1.1 Create Model Module Structure
```
src/models/glm_ocr/
├── mod.rs       # Module exports
├── config.rs    # Model configuration structs
├── model.rs     # Vision encoder, projector, language model
├── processor.rs # Image preprocessing and tokenization
└── generate.rs  # GenerateModel trait implementation
```

#### 1.2 Configuration (`config.rs`)

```rust
// Main config combining all components
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
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
    pub torch_dtype: String,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct GlmOCRVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct GlmOCRProjectorConfig {
    pub hidden_size: usize,
    pub projector_hidden_act: String,
    pub num_queries: usize,  // Number of resampler queries
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
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
    pub hidden_act: String,
    pub use_cache: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct GlmOCRGenerationConfig {
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    pub do_sample: bool,
    pub eos_token_id: Vec<usize>,
    pub top_p: f32,
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
}
```

#### 1.3 Model Architecture (`model.rs`)

**Vision Encoder (ViT)**:
```rust
pub struct GlmOCRVisionEncoder {
    patch_embed: Conv2d,
    pos_embed: Tensor,
    blocks: Vec<GlmOCRVisionBlock>,
    post_layernorm: LayerNorm,
}
```

**Vision Block**:
```rust
pub struct GlmOCRVisionBlock {
    norm1: LayerNorm,
    attn: GlmOCRVisionAttention,
    norm2: LayerNorm,
    mlp: TwoLinearMLP,
}
```

**Projector (Resampler/Perceiver)**:
```rust
pub struct GlmOCRProjector {
    query_embed: Tensor,  // Learnable queries
    proj: Linear,
    norm: LayerNorm,
}
```

**Language Model (GLM-style)**:
```rust
pub struct GlmOCRLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<GlmOCRDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}
```

**Full Model**:
```rust
pub struct GlmOCRModel {
    vision_encoder: GlmOCRVisionEncoder,
    projector: GlmOCRProjector,
    language_model: GlmOCRLanguageModel,
}
```

#### 1.4 Processor (`processor.rs`)

```rust
pub struct GlmOCRProcessor {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    image_size: usize,
    device: Device,
    dtype: DType,
}

impl GlmOCRProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self>;
    
    pub fn process_image(&self, image_path: &str) -> Result<Tensor>;
    
    pub fn process_info(
        &self,
        mes: &ChatCompletionParameters,
        tokenizer: &TokenizerModel,
    ) -> Result<ProcessedInput>;
}

pub struct ProcessedInput {
    pub input_ids: Tensor,
    pub pixel_values: Tensor,
    pub image_mask: Tensor,
}
```

#### 1.5 Generate Model (`generate.rs`)

```rust
pub struct GlmOCRGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmOCRProcessor,
    model: GlmOCRModel,
    device: Device,
    eos_token_id: u32,
    generation_config: GlmOCRGenerationConfig,
    model_name: String,
}

impl<'a> GenerateModel for GlmOCRGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
    fn generate_stream(...) -> Result<Box<dyn Stream<...>>>;
}
```

### Phase 2: Integration with Model System

#### 2.1 Update `src/models/mod.rs`

1. Add module declaration:
```rust
pub mod glm_ocr;
```

2. Add WhichModel variant:
```rust
#[value(name = "glm-ocr", hide = true)]
GlmOCR,
```

3. Add to `ModelInstance` enum:
```rust
GlmOCR(GlmOCRGenerateModel<'a>),
```

4. Add match arms in `load_model()` and `GenerateModel` impl

### Phase 3: API Layer

No changes needed - uses existing `/chat/completions` endpoint via `GenerateModel` trait.

### Phase 4: CLI Integration

#### 4.1 Create Exec Module (`src/exec/glm_ocr.rs`)

```rust
pub struct GlmOCRExec;

impl ExecModel for GlmOCRExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        // Parse image path from input
        // Run OCR inference
        // Display or save results
    }
}
```

#### 4.2 Update `src/exec/mod.rs`
```rust
pub mod glm_ocr;
```

#### 4.3 Update `src/main.rs`

1. Add model ID:
```rust
WhichModel::GlmOCR => "ZhipuAI/GLM-OCR",
```

2. Add to `run_list()`

3. Add to `run_run()` match arm:
```rust
WhichModel::GlmOCR => {
    use aha::exec::glm_ocr::GlmOCRExec;
    GlmOCRExec::run(&input, output.as_deref(), &weight_path)?;
}
```

### Phase 5: Testing

#### 5.1 Create Integration Test (`tests/test_glm_ocr.rs`)

```rust
#[test]
#[ignore = "requires model download"]
fn glm_ocr_load() {
    // Test model loading
}

#[test]
#[ignore = "requires model download"]
fn glm_ocr_image_recognition() {
    // Test basic OCR on an image
}
```

## Key Design Decisions

1. **Vision Encoder**: Use standard ViT architecture with patch embedding + transformer blocks
2. **Projector**: Use resampler pattern (learnable queries + cross-attention) to compress image tokens
3. **Language Model**: GLM-style architecture similar to GlmASRNano but for text generation
4. **Image Processing**: Standard normalization and resizing to fixed size
5. **Token Integration**: Image tokens inserted at special token positions in the prompt

## API Usage Examples

### HTTP API
```bash
POST /chat/completions
{
  "model": "glm-ocr",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.png"},
        {"type": "text", "text": "Extract all text from this image."}
      ]
    }
  ]
}
```

### CLI
```bash
# Run OCR on an image
aha run -m glm-ocr -i "path/to/image.png"

# With custom prompt
aha run -m glm-ocr -i "path/to/image.png" -i "Extract the table data"
```

## Implementation Checklist

- [ ] Create `src/models/glm_ocr/` module
- [ ] Implement `config.rs` with all config structs
- [ ] Implement `model.rs` with vision encoder, projector, and language model
- [ ] Implement `processor.rs` for image preprocessing
- [ ] Implement `generate.rs` with `GenerateModel` trait
- [ ] Update `src/models/mod.rs` with WhichModel variant
- [ ] Create `src/exec/glm_ocr.rs`
- [ ] Update `src/exec/mod.rs`
- [ ] Update `src/main.rs` with model IDs and routing
- [ ] Create integration tests
- [ ] Update documentation

## Notes

- The actual architecture details should be verified from the official GLM-OCR config.json
- Image token handling may vary - check if using special tokens or direct pixel input
- The projector architecture (resampler vs MLP) needs to be confirmed
- RoPE scaling configuration should match the original implementation
