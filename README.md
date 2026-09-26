<h1 align="center">Zephyr</h1>

<p align="center">
  <strong>LLM inference, built from scratch in C++ and CUDA.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-SM80%2B-76B900?style=flat-square" alt="CUDA SM80+">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square" alt="Python 3.10+">
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

Zephyr is an LLM inference engine built from scratch in **C++20 and CUDA** as a hands-on learning project.
It uses its own lightweight tensor library ([TTL](deps/ttl)) and has **no PyTorch dependency**.
Python provides a thin wrapper for tokenization and serving; the inference engine runs entirely in C++.

## Features

| Feature | What it provides |
| :--- | :--- |
| **Tensor parallelism** | Single-node, multi-GPU inference in one process, with a worker thread per GPU. |
| **Paged KV cache** | Prefix caching and automatic GPU memory budgeting. |
| **Continuous batching** | Dynamic request batching, chunked prefill, and preemption under cache pressure. |
| **Python APIs** | Synchronous and asynchronous generation, streaming, and request cancellation. |
| **OpenAI-compatible serving** | Text and chat completion endpoints with streaming responses. |

## Quick Start

Requires Linux, an NVIDIA GPU with compute capability 8.0 or newer, Python 3.10+, a C++20 compiler,
CUDA Toolkit, NCCL, CMake 3.24+, and [uv](https://docs.astral.sh/uv/getting-started/installation/).

> [!NOTE]
> Currently supported models: **Qwen3 dense models**, in FP16, BF16, and FP32. More architectures are on the roadmap.

### 1. Install

Clone and install from source (this compiles the C++/CUDA engine):

```bash
git clone https://github.com/Poseidon-fan/zephyr.git
cd zephyr
uv sync --extra server
```

### 2. Start the server

Launch Qwen3-0.6B on one GPU:

```bash
uv run --no-sync zephyr serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3 \
  --max-model-len 4096
```

Model files are downloaded and cached automatically if needed; a local model directory works too.
For two GPUs, add `--tensor-parallel-size 2`.

### 3. Send a request

Once the server is ready, send a request from another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

This example disables Qwen3's thinking mode. Add `"stream": true` to the request and `-N` to `curl` for streaming.

<details>
<summary><strong>Use the engine directly from Python</strong></summary>

No HTTP server is needed. Save this as a script and run it with `uv run --no-sync python your_script.py`:

```python
from zephyr import Engine

with Engine("Qwen/Qwen3-0.6B", max_seq_len=4096) as engine:
    output = engine.chat(
        [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
        max_tokens=128,
        chat_template_kwargs={"enable_thinking": False},
    )
    print(output.choices[0].text)
```

`Engine` provides synchronous calls; `AsyncEngine` supports concurrent requests and async streaming.

</details>

<details>
<summary><strong>Common server options</strong></summary>

| Option | Purpose |
| :--- | :--- |
| `--tensor-parallel-size 2` | Run the model across two GPUs. |
| `--gpu-memory-utilization 0.9` | Set the GPU memory budget as a fraction of total device memory. |
| `--max-model-len 4096` | Limit the total context length, including prompt and generated tokens. |
| `--log-level debug` | Show more detailed engine and server logs. |

For all options:

```bash
uv run --no-sync zephyr serve --help
```

</details>

## Roadmap

- [ ] **Code refinement** — Review and polish code developed in collaboration with coding agents,
  focusing on correctness, clarity, and performance.
- [ ] **More models** — Expand model support beyond the current Qwen3 dense architecture.
- [ ] **CUDA Graphs** — Integrate TTL's existing graph capture and replay support into Zephyr,
  addressing buffer stability and capture constraints in the execution path.

---

[License](LICENSE) · [Report an issue](https://github.com/Poseidon-fan/zephyr/issues)
