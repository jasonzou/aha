use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse, ChatMessage,
    ChatMessageContent, ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        glm_ocr::{
            config::{GlmOCRConfig, GlmOCRGenerationConfig},
            model::GlmOCRModel,
            processor::GlmOCRProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct GlmOCRGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmOCRProcessor,
    model: GlmOCRModel,
    device: Device,
    eos_token_id: u32,
    generation_config: GlmOCRGenerationConfig,
    model_name: String,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
}

impl<'a> GlmOCRGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: GlmOCRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = if cfg.torch_dtype.is_empty() {
            "bfloat16"
        } else {
            &cfg.torch_dtype
        };
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = GlmOCRProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = GlmOCRModel::new(vb, cfg.clone())?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: GlmOCRGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;

        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            model,
            device,
            eos_token_id: cfg.eos_token_id,
            generation_config,
            model_name: "glm-ocr".to_string(),
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
        })
    }
}

impl<'a> GenerateModel for GlmOCRGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);

        // Extract image path and prompt from messages
        let (image_path, prompt) = extract_image_and_prompt(&mes)?;

        let processed = self.processor.process_info(
            &image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
        )?;

        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        for _ in 0..sample_len {
            let logits = self.model.forward(
                &input_ids,
                pixel_values.as_ref(),
                image_mask.as_ref(),
                seqlen_offset,
            )?;
            let logits = logits.i((0, seq_len - 1, ..))?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        let response = build_completion_response(res, &self.model_name, Some(num_token));
        Ok(response)
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);

        // Extract image path and prompt from messages
        let (image_path, prompt) = extract_image_and_prompt(&mes)?;

        let processed = self.processor.process_info(
            &image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
        )?;

        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);

        let stream = stream! {
            let mut error_tokens = Vec::new();
            for _ in 0..sample_len {
                let logits = self.model.forward(
                    &input_ids,
                    pixel_values.as_ref(),
                    image_mask.as_ref(),
                    seqlen_offset,
                ).map_err(|e| anyhow!(format!("forward error: {e}")))?;
                let logits = logits.i((0, seq_len - 1, ..)).map_err(|e| anyhow!(format!("index error: {e}")))?.to_dtype(DType::F32).map_err(|e| anyhow!(format!("dtype error: {e}")))?;
                let next_token = logit_processor.sample(&logits).map_err(|e| anyhow!(format!("sample error: {e}")))?;

                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);

                let decoded_token = self.tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("decode error: {e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device).map_err(|e| anyhow!(format!("tensor error: {e}")))?;
                    continue;
                }
                error_tokens.clear();

                let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                yield Ok(chunk);

                if next_token == self.eos_token_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device).map_err(|e| anyhow!(format!("tensor error: {e}")))?;
            }
        };

        Ok(Box::new(Box::pin(stream)))
    }
}

fn extract_image_and_prompt(mes: &ChatCompletionParameters) -> Result<(String, String)> {
    let mut image_path = None;
    let mut prompt = String::new();

    for message in &mes.messages {
        if let ChatMessage::User { content, .. } = message {
            match content {
                ChatMessageContent::ContentPart(part_vec) => {
                    for part in part_vec {
                        match part {
                            ChatMessageContentPart::Image(img_part) => {
                                image_path = Some(img_part.image_url.url.clone());
                            }
                            ChatMessageContentPart::Text(text_part) => {
                                prompt.push_str(&text_part.text);
                            }
                            _ => {}
                        }
                    }
                }
                ChatMessageContent::Text(text) => {
                    prompt.push_str(text);
                }
                _ => {}
            }
        }
    }

    let image_path = image_path.ok_or_else(|| anyhow!("No image provided"))?;
    if prompt.is_empty() {
        prompt = "Extract all text from this image.".to_string();
    }

    Ok((image_path, prompt))
}
