pub mod campplus;
pub mod common;
pub mod feature_extractor;
pub mod fire_red_vad;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3;

use crate::params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use rocket::futures::Stream;

use crate::models::{
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel,
    qwen3::generate::Qwen3GenerateModel,
    qwen3_5::generate::Qwen3_5GenerateModel,
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
    Qwen3_5(Qwen3_5GenerateModel<'a>),
    Qwen3ASR(Qwen3AsrGenerateModel<'a>),
    GlmASRNano(GlmAsrNanoGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3_5(model) => model.generate(mes),
            ModelInstance::Qwen3ASR(model) => model.generate(mes),
            ModelInstance::GlmASRNano(model) => model.generate(mes),
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
            ModelInstance::Qwen3_5(model) => model.generate_stream(mes),
            ModelInstance::Qwen3ASR(model) => model.generate_stream(mes),
            ModelInstance::GlmASRNano(model) => model.generate_stream(mes),
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
    _model_type: crate::models::common::model_mapping::WhichModel,
    _path: &str,
    _device: Option<&Device>,
    _dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    Err(anyhow!("use specific ASR model loaders instead"))
}

pub fn load_qwen3_model<'a>(
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    let model = Qwen3GenerateModel::init(path, device, dtype)?;
    Ok(ModelInstance::Qwen3(model))
}

pub fn load_qwen3_5_model<'a>(
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    let model = Qwen3_5GenerateModel::init(path, device, dtype)?;
    Ok(ModelInstance::Qwen3_5(model))
}

pub fn load_qwen3_asr_model<'a>(
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    let model = Qwen3AsrGenerateModel::init(path, device, dtype)?;
    Ok(ModelInstance::Qwen3ASR(model))
}

pub fn load_glm_asr_nano_model<'a>(
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    let model = GlmAsrNanoGenerateModel::init(path, device, dtype)?;
    Ok(ModelInstance::GlmASRNano(model))
}

pub fn load_fun_asr_nano_model<'a>(
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
) -> Result<ModelInstance<'a>> {
    let model = FunAsrNanoGenerateModel::init(path, device, dtype)?;
    Ok(ModelInstance::FunASRNano(model))
}