use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::qwen3_reranker::model::Qwen3RerankerModel;
use crate::utils::get_file_path;

pub struct Qwen3RerankerExec;

impl ExecModel for Qwen3RerankerExec {
    fn run(input: &[String], _output: Option<&str>, weight_path: &str) -> Result<()> {
        // Parse input: first argument is query, rest are documents
        if input.is_empty() {
            return Err(anyhow::anyhow!(
                "Input required: first argument is query, subsequent arguments are documents"
            ));
        }

        let query = &input[0];
        let query_text = if query.starts_with("file://") {
            let path = get_file_path(query)?;
            std::fs::read_to_string(path)?.trim().to_string()
        } else {
            query.clone()
        };

        let documents: Vec<String> = if input.len() > 1 {
            input[1..]
                .iter()
                .map(|doc| {
                    if doc.starts_with("file://") {
                        let path = get_file_path(doc).ok()?;
                        std::fs::read_to_string(path)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        Some(doc.clone())
                    }
                })
                .filter_map(|x| x)
                .collect()
        } else {
            return Err(anyhow::anyhow!(
                "At least one document required for reranking"
            ));
        };

        if documents.is_empty() {
            return Err(anyhow::anyhow!("No valid documents provided"));
        }

        println!("Query: {}", query_text);
        println!("Documents to rerank: {}", documents.len());
        println!();

        let i_start = Instant::now();
        let mut model = Qwen3RerankerModel::init(weight_path, None, None)?;
        println!("Time elapsed in load model: {:?}", i_start.elapsed());

        let i_start = Instant::now();
        let scores = model.compute_scores(&query_text, &documents)?;
        println!("Time elapsed in reranking: {:?}", i_start.elapsed());
        println!();

        // Sort results by score
        let mut indexed_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("Results (sorted by relevance):");
        println!("{:<5} {:<12} {}", "Rank", "Score", "Document");
        println!("{}", "-".repeat(80));
        for (rank, (idx, score)) in indexed_scores.iter().enumerate() {
            let doc = &documents[*idx];
            let doc_preview = if doc.len() > 60 {
                format!("{}...", &doc[..60])
            } else {
                doc.clone()
            };
            println!("{:<5} {:.6}     {}", rank + 1, score, doc_preview);
        }

        Ok(())
    }
}
