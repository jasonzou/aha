# 快速入门

欢迎使用 AHA！本指南将帮助您快速上手 ASR 模型。

## 快速开始（5 分钟）

### 1. 查看可用模型

```bash
aha list
```

### 2. 下载第一个模型

```bash
# 下载 ASR 模型开始
aha download -m Qwen/Qwen3-ASR-0.6B
```

### 3. 启动服务

```bash
# 启动 HTTP API 服务器
aha cli -m Qwen/Qwen3-ASR-0.6B
```

服务将在 `http://127.0.0.1:10100` 上启动

### 4. 发起第一个 API 调用

在新终端中：

```bash
curl http://127.0.0.1:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-0.6B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "转写这段音频。"},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ],
    "stream": false
  }'
```

## 基本概念

### 什么是 AHA？

AHA 是一个本地 ASR 推理引擎，具有以下特点：
- 在您的机器上运行语音识别模型（无需云 API）
- 支持多种 ASR 模型（中文/英文）
- 提供 OpenAI 兼容的 API
- 模型下载后可离线工作

### 模型类别

| 类别 | 描述 | 示例模型 |
|------|------|----------|
| **ASR** | 语音转文本 | GLM-ASR、Fun-ASR、Qwen3-ASR |

### CLI 命令

| 命令 | 用途 |
|------|------|
| `aha cli` | 下载模型并启动服务 |
| `aha serv` | 使用现有模型启动服务 |
| `aha download` | 仅下载模型 |
| `aha run` | 直接推理，无需服务器 |
| `aha list` | 列出可用模型 |

## 常见工作流程

### 语音识别（ASR）

```bash
# 启动 ASR 模型
aha cli -m Qwen/Qwen3-ASR-0.6B

# 转写音频
curl http://127.0.0.1:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-0.6B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "转写这段音频。"},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ],
    "stream": false
  }'
```

### 直接推理（无需服务器）

```bash
# 直接运行推理，无需启动 HTTP 服务器
aha run -m Qwen/Qwen3-ASR-0.6B \
  -i "audio.wav" \
  --weight-path ~/.aha/Qwen/Qwen3-ASR-0.6B
```

## 配置选项

### 更改端口

```bash
# 使用端口 8080 而不是默认的 10100
aha cli -m Qwen/Qwen3-ASR-0.6B -p 8080
```

### 绑定到所有接口

```bash
# 允许外部访问（请谨慎使用）
aha cli -m Qwen/Qwen3-ASR-0.6B -a 0.0.0.0 -p 8080
```

### 使用本地模型

```bash
# 跳过下载，使用现有模型
aha serv -m Qwen/Qwen3-ASR-0.6B \
  --weight-path /path/to/model \
  -p 8080
```

### 自定义保存目录

```bash
# 将模型下载到特定目录
aha download -m Qwen/Qwen3-ASR-0.6B -s /data/models
```

## 模型选择指南

### 语音识别
- **Qwen/Qwen3-ASR-0.6B**：快速、轻量级
- **Qwen/Qwen3-ASR-1.7B**：更高质量
- **ZhipuAI/GLM-ASR-Nano-2512**：快速、准确
- **FunAudioLLM/Fun-ASR-Nano-2512**：适合中文

## 提示与最佳实践

### 1. 使用 GPU 加速

使用 GPU 支持构建以获得更好的性能：
```bash
# NVIDIA GPU
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### 2. 预先下载模型

在网络良好时下载模型：
```bash
aha download -m Qwen/Qwen3-ASR-0.6B
```

稍后在没有网络的情况下使用：
```bash
aha serv -m Qwen/Qwen3-ASR-0.6B --weight-path ~/.aha/Qwen/Qwen3-ASR-0.6B
```

### 3. 管理磁盘空间

模型默认存储在 `~/.aha/` 中。如需要，清理：
```bash
# 检查磁盘使用情况
du -sh ~/.aha/*

# 删除旧模型
rm -rf ~/.aha/old-model-name
```

### 4. 监控资源

对于大型模型，监控您的资源：
```bash
# Linux
htop
nvidia-smi  # 对于 NVIDIA GPU

# macOS
活动监视器
```

## 故障排除

### 端口已被占用

```bash
# 使用不同的端口
aha cli -m Qwen/Qwen3-ASR-0.6B -p 8080
```

### 模型下载失败

```bash
# 重试更多次数
aha download -m Qwen/Qwen3-ASR-0.6B --download-retries 5
```

### 内存不足

```bash
# 使用更小的模型
aha cli -m Qwen/Qwen3-ASR-0.6B
```

## 后续步骤

1. 探索 [API 参考](./api.zh-CN.md) 了解详细的端点文档
2. 阅读 [CLI 参考](./cli.zh-CN.md) 了解所有命令选项
3. 查看 [架构与设计](./concepts.zh-CN.md) 了解 AHA 的工作原理
4. 如果您想贡献，请参阅 [开发指南](./development.zh-CN.md)

## 示例仓库

更多示例，请查看仓库中的 [tests](../tests/) 目录。

## 另见

- [API 参考](./api.zh-CN.md) - 完整的 API 文档
- [CLI 参考](./cli.zh-CN.md) - 命令行参考
- [安装指南](./installation.zh-CN.md) - 安装说明
- [开发指南](./development.zh-CN.md) - 贡献指南