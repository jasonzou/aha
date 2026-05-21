# Getting Started

Welcome to AHA! This guide will help you get up and running quickly with ASR models.

## Quick Start (5 Minutes)

### 1. Check Available Models

```bash
aha list
```

### 2. Download Your First Model

```bash
# Download an ASR model to start
aha download -m Qwen/Qwen3-ASR-0.6B
```

### 3. Start the Service

```bash
# Start the HTTP API server
aha cli -m Qwen/Qwen3-ASR-0.6B
```

The service will start on `http://127.0.0.1:10100`

### 4. Make Your First API Call

In a new terminal:

```bash
curl http://127.0.0.1:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-0.6B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Transcribe this audio."},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ],
    "stream": false
  }'
```

## Basic Concepts

### What is AHA?

AHA is a local ASR inference engine that:
- Runs speech recognition models on your machine (no cloud API)
- Supports multiple ASR models (Chinese/English)
- Provides an OpenAI-compatible API
- Works offline once models are downloaded

### Model Categories

| Category | Description | Example Models |
|----------|-------------|----------------|
| **ASR** | Speech-to-text | GLM-ASR, Fun-ASR, Qwen3-ASR |

### CLI Commands

| Command | Purpose |
|---------|---------|
| `aha cli` | Download model and start service |
| `aha serv` | Start service with existing model |
| `aha download` | Download model only |
| `aha run` | Direct inference without server |
| `aha list` | List available models |

## Common Workflows

### Speech Recognition (ASR)

```bash
# Start an ASR model
aha cli -m Qwen/Qwen3-ASR-0.6B

# Transcribe audio
curl http://127.0.0.1:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-0.6B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Transcribe this audio."},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ],
    "stream": false
  }'
```

### Direct Inference (Without Server)

```bash
# Run inference directly without starting HTTP server
aha run -m Qwen/Qwen3-ASR-0.6B \
  -i "audio.wav" \
  --weight-path ~/.aha/Qwen/Qwen3-ASR-0.6B
```

## Configuration Options

### Change Port

```bash
# Use port 8080 instead of default 10100
aha cli -m Qwen/Qwen3-ASR-0.6B -p 8080
```

### Bind to All Interfaces

```bash
# Allow external access (use with caution)
aha cli -m Qwen/Qwen3-ASR-0.6B -a 0.0.0.0 -p 8080
```

### Use Local Model

```bash
# Skip download, use existing model
aha serv -m Qwen/Qwen3-ASR-0.6B \
  --weight-path /path/to/model \
  -p 8080
```

### Custom Save Directory

```bash
# Download model to specific directory
aha download -m Qwen/Qwen3-ASR-0.6B -s /data/models
```

## Model Selection Guide

### For Speech Recognition
- **Qwen/Qwen3-ASR-0.6B**: Fast, lightweight
- **Qwen/Qwen3-ASR-1.7B**: Better quality
- **ZhipuAI/GLM-ASR-Nano-2512**: Fast, accurate
- **FunAudioLLM/Fun-ASR-Nano-2512**: Good for Chinese

## Tips & Best Practices

### 1. Use GPU Acceleration

Build with GPU support for better performance:
```bash
# NVIDIA GPUs
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### 2. Pre-download Models

Download models when you have good internet:
```bash
aha download -m Qwen/Qwen3-ASR-0.6B
```

Then use them later without internet:
```bash
aha serv -m Qwen/Qwen3-ASR-0.6B --weight-path ~/.aha/Qwen/Qwen3-ASR-0.6B
```

### 3. Manage Disk Space

Models are stored in `~/.aha/` by default. Clean up if needed:
```bash
# Check disk usage
du -sh ~/.aha/*

# Remove old models
rm -rf ~/.aha/old-model-name
```

### 4. Monitor Resources

For large models, monitor your resources:
```bash
# Linux
htop
nvidia-smi  # For NVIDIA GPUs

# macOS
Activity Monitor
```

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
aha cli -m Qwen/Qwen3-ASR-0.6B -p 8080
```

### Model Download Failed

```bash
# Retry with more attempts
aha download -m Qwen/Qwen3-ASR-0.6B --download-retries 5
```

### Out of Memory

```bash
# Use a smaller model
aha cli -m Qwen/Qwen3-ASR-0.6B
```

## Next Steps

1. Explore the [API Reference](./api.md) for detailed endpoint documentation
2. Read the [CLI Reference](./cli.md) for all command options
3. Check [Architecture & Design](./concepts.md) to understand how AHA works
4. See [Development](./development.md) if you want to contribute

## Examples Repository

For more examples, check out the [tests](../tests/) directory in the repository.

## See Also

- [API Reference](./api.md) - Complete API documentation
- [CLI Reference](./cli.md) - Command-line reference
- [Installation Guide](./installation.md) - Installation instructions
- [Development Guide](./development.md) - Contributing guide