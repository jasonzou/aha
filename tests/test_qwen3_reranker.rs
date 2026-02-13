use std::time::Instant;

use aha::models::{RerankModel, RerankParameters, qwen3_reranker::model::Qwen3RerankerModel};

const MODEL_PATH: &str = "~/.aha/Qwen/Qwen3-Reranker-0.6B";

fn get_model_path() -> String {
    let path = MODEL_PATH.replace("~", &std::env::var("HOME").unwrap_or_default());
    path
}

#[test]
#[ignore = "requires model download"]
fn qwen3_reranker_0_6b_load() {
    let path = get_model_path();
    let i_start = Instant::now();
    let model = Qwen3RerankerModel::init(&path, None, None);
    println!("Time elapsed in load model: {:?}", i_start.elapsed());
    assert!(model.is_ok(), "Failed to load model: {:?}", model.err());
}

#[test]
#[ignore = "requires model download"]
fn qwen3_reranker_0_6b_rerank() {
    let path = get_model_path();
    let mut model = Qwen3RerankerModel::init(&path, None, None).expect("Failed to load model");

    let query = "What is machine learning?".to_string();
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.".to_string(),
        "The weather today is sunny with a high of 75 degrees.".to_string(),
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.".to_string(),
    ];

    let params = RerankParameters {
        query,
        documents: documents.clone(),
        model: "qwen3-reranker-0.6b".to_string(),
        top_k: usize::MAX,
        return_documents: true,
    };

    let i_start = Instant::now();
    let result = model.rerank(params);
    println!("Time elapsed in reranking: {:?}", i_start.elapsed());

    assert!(result.is_ok(), "Reranking failed: {:?}", result.err());
    let response = result.unwrap();

    println!("Reranking results:");
    for (rank, res) in response.results.iter().enumerate() {
        println!(
            "Rank {}: Index {} - Score {:.4}",
            rank + 1,
            res.index,
            res.relevance_score
        );
    }

    // Check that we got results for all documents
    assert_eq!(response.results.len(), documents.len());

    // Check that results are sorted by score (descending)
    for i in 1..response.results.len() {
        assert!(
            response.results[i - 1].relevance_score >= response.results[i].relevance_score,
            "Results should be sorted by relevance score in descending order"
        );
    }

    // The first document about machine learning should have a high score
    // and likely be ranked first or second
    let ml_doc_score = response
        .results
        .iter()
        .find(|r| r.index == 0)
        .map(|r| r.relevance_score)
        .unwrap_or(0.0);
    assert!(
        ml_doc_score > 0.5,
        "Machine learning document should have high relevance score"
    );
}

#[test]
#[ignore = "requires model download"]
fn qwen3_reranker_0_6b_top_k() {
    let path = get_model_path();
    let mut model = Qwen3RerankerModel::init(&path, None, None).expect("Failed to load model");

    let query = "What is machine learning?".to_string();
    let documents = vec![
        "Document 1 about ML.".to_string(),
        "Document 2 about weather.".to_string(),
        "Document 3 about deep learning.".to_string(),
        "Document 4 about cooking.".to_string(),
        "Document 5 about AI.".to_string(),
    ];

    let params = RerankParameters {
        query,
        documents,
        model: "qwen3-reranker-0.6b".to_string(),
        top_k: 2,
        return_documents: false,
    };

    let result = model.rerank(params);
    assert!(result.is_ok());
    let response = result.unwrap();

    // Check that top_k limit is applied
    assert_eq!(
        response.results.len(),
        2,
        "Should return only top_k results"
    );
}
