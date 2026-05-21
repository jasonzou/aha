use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "Qwen/Qwen3-ASR-0.6B")]
    Qwen3ASR0_6B,
    #[value(name = "Qwen/Qwen3-ASR-1.7B")]
    Qwen3ASR1_7B,
    #[value(name = "ZhipuAI/GLM-ASR-Nano-2512")]
    GlmASRNano2512,
    #[value(name = "FunAudioLLM/Fun-ASR-Nano-2512")]
    FunASRNano2512,
}

impl WhichModel {
    /// Get the model ID for this model variant
    pub fn as_string(&self) -> String {
        self.to_possible_value()
            .expect("not exists")
            .get_name()
            .to_string()
    }

    /// Checks if the model is in GGUF format
    ///
    /// Returns true if the model ID contains "gguf", false otherwise
    pub fn is_gguf(&self) -> bool {
        let model_id = self.as_string();
        model_id.to_lowercase().contains("gguf")
    }

    /// Checks if the model is in ONNX format
    ///
    /// Returns true if the model ID contains "onnx", false otherwise
    pub fn is_onnx(&self) -> bool {
        let model_id = self.as_string();
        model_id.to_lowercase().contains("onnx")
    }

    /// Get the WhichModel enum list
    pub fn model_list() -> Vec<Self> {
        WhichModel::value_variants().to_vec()
    }

    /// Extracts the model owner/organization from the model ID
    ///
    /// Splits the model ID string on '/' and returns the first part which typically represents
    /// the organization or user who owns the model in Hugging Face format (e.g., "Qwen" from "Qwen/Qwen3-0.6B")
    /// Returns "none" if the model ID doesn't contain a '/' separator
    pub fn model_owner(&self) -> String {
        let name = self.as_string();
        let names: Vec<&str> = name.split("/").collect();
        if names.len() < 2 {
            "none".to_string()
        } else {
            names.first().map_or("none", |&s| s).to_string()
        }
    }

    /// Get the model type category for this model variant
    pub fn model_type(self) -> &'static str {
        match self {
            WhichModel::Qwen3ASR0_6B
            | WhichModel::Qwen3ASR1_7B
            | WhichModel::GlmASRNano2512
            | WhichModel::FunASRNano2512 => "asr",
        }
    }
}
