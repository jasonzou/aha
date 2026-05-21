# 支持的模型

aha 支持一系列先进的 ASR（自动语音识别）模型。

```shell
可用模型：

Model ID                                 Owner                type       Download
--------------------------------------------------------------------------------
Qwen/Qwen3-ASR-0.6B                      Qwen                 asr          ✔
Qwen/Qwen3-ASR-1.7B                      Qwen                 asr
ZhipuAI/GLM-ASR-Nano-2512                ZhipuAI              asr          ✔
FunAudioLLM/Fun-ASR-Nano-2512            FunAudioLLM          asr          ✔
```

## 语音识别 (ASR)

| 模型 | 参数量 | 语言 | 模型id | 开源协议 |
|------|--------|------|-----|---------|
| **Fun-ASR-Nano-2512** | - | 中/英 | FunAudioLLM/Fun-ASR-Nano-2512 | 未标明 |
| **GLM-ASR-Nano-2512** | - | 中/英 | ZhipuAI/GLM-ASR-Nano-2512 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR** | 0.6B <br> 1.7B | 中/英 | Qwen/Qwen3-ASR-0.6B <br> Qwen/Qwen3-ASR-1.7B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 模型来源

模型来源：

- [Hugging Face](https://huggingface.co) - 主模型中心
- [ModelScope](https://modelscope.cn) - 中文模型中心

## 添加新模型

参见 [开发指南](./development.zh-CN.md) 了解添加新模型集成的说明。

## 许可证合规

**重要提示**：每个模型都有自己的许可证。在生产环境使用前请查看模型许可证。

- **Apache 2.0**: 宽松许可，支持商业使用
- **MIT**: 宽松许可，支持商业使用

在生产环境部署前，请务必验证许可证条款。

## 模型更新

模型不定期更新。

## 性能基准

CPU (M1 Pro) 上的近似推理速度：

| 模型 | 任务 | Tokens/秒 |
|------|------|-----------|
| Qwen3-ASR-0.6B | ASR | 200-500x |

*基准测试因硬件和输入大小而异。*