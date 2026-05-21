# Supported Models

aha supports a collection of state-of-the-art ASR (Automatic Speech Recognition) models.

```shell
Available models:

Model ID                                 Owner                type       Download
--------------------------------------------------------------------------------
Qwen/Qwen3-ASR-0.6B                      Qwen                 asr          ✔
Qwen/Qwen3-ASR-1.7B                      Qwen                 asr
ZhipuAI/GLM-ASR-Nano-2512                ZhipuAI              asr          ✔
FunAudioLLM/Fun-ASR-Nano-2512            FunAudioLLM          asr          ✔
```

## Speech Recognition (ASR)

| Model | Parameters | Language | Model Id | License |
|-------|-----------|----------|----------|---------|
| **Fun-ASR-Nano-2512** | - | Chinese/English | FunAudioLLM/Fun-ASR-Nano-2512 | Not Specified |
| **GLM-ASR-Nano-2512** | - | Chinese/English | ZhipuAI/GLM-ASR-Nano-2512 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR** | 0.6B <br> 1.7B | Chinese/English | Qwen/Qwen3-ASR-0.6B <br> Qwen/Qwen3-ASR-1.7B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Model Sources

Models are sourced from:

- [Hugging Face](https://huggingface.co) - Primary model hub
- [ModelScope](https://modelscope.cn) - Chinese model hub

## Adding New Models

See [Development Guide](./development.md) for instructions on adding new model integrations.

## License Compliance

**Important**: Each model has its own license. Please review the model's license before use in production.

- **Apache 2.0**: Permissive, commercial-friendly
- **MIT**: Permissive, commercial-friendly

Always verify license terms before deployment in production environments.

## Model Updates

Models updated from time to time.

## Performance Benchmarks

Approximate inference speeds on CPU (M1 Pro):

| Model | Task | Tokens/sec |
|-------|------|------------|
| Qwen3-ASR-0.6B | ASR | 200-500x |

*Benchmarks vary by hardware and input size.*