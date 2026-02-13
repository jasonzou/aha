use std::time::Instant;

use aha::models::glm_ocr::generate::GlmOCRGenerateModel;

const MODEL_PATH: &str = "~/.aha/ZhipuAI/GLM-OCR";

fn get_model_path() -> String {
    let path = MODEL_PATH.replace("~", &std::env::var("HOME").unwrap_or_default());
    path
}

#[test]
#[ignore = "requires model download"]
fn glm_ocr_load() {
    let path = get_model_path();
    let i_start = Instant::now();
    let model = GlmOCRGenerateModel::init(&path, None, None);
    println!("Time elapsed in load model: {:?}", i_start.elapsed());
    assert!(model.is_ok(), "Failed to load model: {:?}", model.err());
}

#[test]
#[ignore = "requires model download"]
fn glm_ocr_image_recognition() {
    let path = get_model_path();
    let mut model = GlmOCRGenerateModel::init(&path, None, None).expect("Failed to load model");

    // Create a simple test image path (user should provide actual image)
    let test_image = std::env::var("GLM_OCR_TEST_IMAGE").unwrap_or_else(|_| {
        eprintln!("Warning: GLM_OCR_TEST_IMAGE not set, using dummy path");
        "/tmp/test_image.png".to_string()
    });

    let prompt = "Extract all text from this image.";
    let content = format!(
        r#"[{{"type": "image", "image": "{}"}}, {{"type": "text", "text": "{}"}}]"#,
        test_image, prompt
    );

    let mes = aha_openai_dive::v1::resources::chat::ChatCompletionParameters {
        model: "glm-ocr".to_string(),
        messages: vec![aha_openai_dive::v1::resources::chat::ChatMessage {
            role: aha_openai_dive::v1::resources::chat::Role::User,
            content: Some(content),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        top_p: None,
        n: None,
        stop: None,
        max_tokens: Some(512),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        seed: None,
        stream: Some(false),
        response_format: None,
        tools: None,
        tool_choice: None,
        metadata: None,
    };

    let i_start = Instant::now();
    let result = model.generate(mes);
    println!("Time elapsed in OCR: {:?}", i_start.elapsed());

    assert!(result.is_ok(), "OCR failed: {:?}", result.err());
    let response = result.unwrap();

    println!("Extracted text:");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    // Basic validation - response should not be empty
    assert!(
        !response.choices[0]
            .message
            .content
            .as_ref()
            .unwrap()
            .is_empty(),
        "OCR response should not be empty"
    );
}
