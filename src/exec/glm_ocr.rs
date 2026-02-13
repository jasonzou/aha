use std::path::PathBuf;
use std::time::Instant;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
    ChatMessageImageContentPart, ChatMessageTextContentPart, ImageUrlType,
};
use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::glm_ocr::generate::GlmOCRGenerateModel;
use crate::utils::get_file_path;

pub struct GlmOCRExec;

impl ExecModel for GlmOCRExec {
    fn run(input: &[String], _output: Option<&str>, weight_path: &str) -> Result<()> {
        if input.is_empty() {
            return Err(anyhow::anyhow!(
                "Input required: provide image path and optional prompt"
            ));
        }

        // First argument is image path
        let image_path = &input[0];
        let image_path: PathBuf = if image_path.starts_with("file://") {
            get_file_path(image_path)?
        } else {
            image_path.clone().into()
        };

        // Optional second argument is prompt
        let prompt = if input.len() > 1 {
            input[1].clone()
        } else {
            "Extract all text from this image.".to_string()
        };

        println!("Image: {}", image_path.display());
        println!("Prompt: {}", prompt);
        println!();

        let i_start = Instant::now();
        let mut model = GlmOCRGenerateModel::init(weight_path, None, None)?;
        println!("Time elapsed in load model: {:?}", i_start.elapsed());

        // Build ChatCompletionParameters
        let image_url = format!("file://{}", image_path.display());
        let mes = ChatCompletionParameters {
            model: "glm-ocr".to_string(),
            messages: vec![ChatMessage::User {
                content: ChatMessageContent::ContentPart(vec![
                    ChatMessageContentPart::Image(ChatMessageImageContentPart {
                        r#type: "image".to_string(),
                        image_url: ImageUrlType {
                            url: image_url,
                            detail: None,
                        },
                    }),
                    ChatMessageContentPart::Text(ChatMessageTextContentPart {
                        r#type: "text".to_string(),
                        text: prompt,
                    }),
                ]),
                name: None,
            }],
            max_tokens: Some(1024),
            stream: Some(false),
            ..Default::default()
        };

        let i_start = Instant::now();
        let result = model.generate(mes)?;
        println!("Time elapsed in OCR: {:?}", i_start.elapsed());
        println!();

        println!("Extracted Text:");
        if let ChatMessage::Assistant {
            content: Some(ChatMessageContent::Text(text)),
            ..
        } = &result.choices[0].message
        {
            println!("{}", text);
        } else {
            println!("No text content found in response");
        }

        Ok(())
    }
}
