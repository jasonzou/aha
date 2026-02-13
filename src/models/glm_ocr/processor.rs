use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::GlmOCRPreprocessorConfig;
use crate::tokenizer::TokenizerModel;
use crate::utils::img_utils::get_image;

pub struct GlmOCRProcessor {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    image_size: usize,
    device: Device,
    dtype: DType,
}

impl GlmOCRProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let config_path = format!("{}/preprocessor_config.json", path);
        let config: GlmOCRPreprocessorConfig = if std::path::Path::new(&config_path).exists() {
            serde_json::from_slice(&std::fs::read(&config_path)?)?
        } else {
            // Default config
            GlmOCRPreprocessorConfig {
                image_mean: vec![0.48145466, 0.4578275, 0.40821073],
                image_std: vec![0.26862954, 0.26130258, 0.27577711],
                image_size: 448,
            }
        };

        Ok(Self {
            image_mean: config.image_mean,
            image_std: config.image_std,
            image_size: config.image_size,
            device: device.clone(),
            dtype,
        })
    }

    pub fn process_image(&self, image_path: &str) -> Result<Tensor> {
        let img = get_image(image_path)?;
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Lanczos3,
        );

        let img = img.to_rgb8();
        let pixels: Vec<f32> = img
            .pixels()
            .flat_map(|p| {
                vec![
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();

        let tensor = Tensor::from_vec(pixels, (self.image_size, self.image_size, 3), &self.device)?;
        let tensor = tensor.permute((2, 0, 1))?; // HWC -> CHW

        // Normalize
        let mean = Tensor::new(self.image_mean.clone(), &self.device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(self.image_std.clone(), &self.device)?.reshape((3, 1, 1))?;
        let tensor = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // Add batch dimension
        let tensor = tensor.unsqueeze(0)?.to_dtype(self.dtype)?;
        Ok(tensor)
    }

    pub fn process_info(
        &self,
        image_path: &str,
        prompt: &str,
        tokenizer: &TokenizerModel,
        image_token_id: u32,
        image_start_token_id: u32,
        image_end_token_id: u32,
    ) -> Result<ProcessedInput> {
        let pixel_values = self.process_image(image_path)?;

        // Build prompt with image tokens
        // Format: <image_start><image_token>...<image_token><image_end>prompt
        let num_image_tokens = 256; // Number of image tokens (should match projector num_queries)
        let mut input_ids_vec = vec![image_start_token_id];
        for _ in 0..num_image_tokens {
            input_ids_vec.push(image_token_id);
        }
        input_ids_vec.push(image_end_token_id);

        // Encode text prompt
        let text_ids = tokenizer.text_encode(prompt.to_string(), &self.device)?;
        let text_ids_vec = text_ids.to_vec1::<u32>()?;
        input_ids_vec.extend(text_ids_vec);

        let input_ids = Tensor::from_vec(
            input_ids_vec.clone(),
            (1, input_ids_vec.len()),
            &self.device,
        )?;

        // Create image mask (1 for image tokens, 0 for text)
        let mut image_mask_vec = vec![0u32; input_ids_vec.len()];
        for i in 1..=num_image_tokens {
            image_mask_vec[i] = 1;
        }
        let image_mask = Tensor::from_vec(image_mask_vec, (1, input_ids_vec.len()), &self.device)?;

        Ok(ProcessedInput {
            input_ids,
            pixel_values,
            image_mask,
        })
    }
}

pub struct ProcessedInput {
    pub input_ids: Tensor,
    pub pixel_values: Tensor,
    pub image_mask: Tensor,
}
