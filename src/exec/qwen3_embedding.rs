use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::{
    EmbedModel, EmbeddingInput, EmbeddingParameters, qwen3_embedding::model::Qwen3EmbeddingModel,
};
use crate::utils::get_file_path;

pub struct Qwen3EmbeddingExec;

impl ExecModel for Qwen3EmbeddingExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let dimensions: Option<usize> = input.get(1).and_then(|s| s.parse().ok());

        let i_start = Instant::now();
        let mut model = Qwen3EmbeddingModel::init(weight_path, None, None)?;
        println!("Time elapsed in load model is: {:?}", i_start.elapsed());

        let params = EmbeddingParameters {
            input: EmbeddingInput::Single(target_text),
            model: "qwen3-embedding".to_string(),
            dimensions,
            encoding_format: "float".to_string(),
        };

        let i_start = Instant::now();
        let result = model.embed(params)?;
        println!("Time elapsed in embedding is: {:?}", i_start.elapsed());

        println!("Embedding dimension: {}", result.data[0].embedding.len());
        println!(
            "First 10 values: {:?}",
            &result.data[0].embedding[..10.min(result.data[0].embedding.len())]
        );

        if let Some(out) = output {
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(out, json)?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}
