pub mod campplus;
pub mod common;
pub mod feature_extractor;
pub mod fire_red_vad;
pub mod fun_asr_nano;
pub mod qwen3;
pub mod qwen3_asr;

use crate::models::common::model_mapping::WhichModel;
use crate::params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use rocket::futures::Stream;

use crate::models::{
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    qwen3::generate::Qwen3GenerateModel,
    qwen3_asr::generate::Qwen3AsrGenerateModel,
};

pub trait GenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
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
    >;
}

pub enum ModelInstance<'a> {
    Qwen3(Qwen3GenerateModel<'a>),
    Qwen3ASR(Qwen3AsrGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3ASR(model) => model.generate(mes),
            ModelInstance::FunASRNano(model) => model.generate(mes),
        }
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
        match self {
            ModelInstance::Qwen3(model) => model.generate_stream(mes),
            ModelInstance::Qwen3ASR(model) => model.generate_stream(mes),
            ModelInstance::FunASRNano(model) => model.generate_stream(mes),
        }
    }
}

impl<'a> ModelInstance<'a> {
    pub fn embedding(&mut self, _input: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!("current model does not support embeddings"))
    }
    pub fn rerank(&mut self, _query: &str, _documents: &[String]) -> Result<Vec<f32>> {
        Err(anyhow!("current model does not support rerank"))
    }
}

pub fn load_model<'a>(
    model_type: crate::models::common::model_mapping::WhichModel,
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    match model_type {
        WhichModel::Qwen3ASR0_6B | WhichModel::Qwen3ASR1_7B => {
            let model = Qwen3AsrGenerateModel::init(path, device, dtype)?;
            Ok(ModelInstance::Qwen3ASR(model))
        }
        WhichModel::GlmASRNano2512 => {
            let model = FunAsrNanoGenerateModel::init(path, device, dtype)?;
            Ok(ModelInstance::FunASRNano(model))
        }
        WhichModel::FunASRNano2512 => {
            let model = FunAsrNanoGenerateModel::init(path, device, dtype)?;
            Ok(ModelInstance::FunASRNano(model))
        }
        _ => {
            Err(anyhow!("model type {:?} not supported in simplified build", model_type))
        }
    }
}