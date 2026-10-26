# NInfer documentation

Start with the [project README](../README.md) for the CUDA 12.8 V100 Duo build, conversion of the
local LM Studio Q4_K_M model, and the 262K-token INT8-KV/MTP3 launcher. It also retains the
published-artifact CLI and HTTP examples for other profiles.

**Current build**: Windows 11 · 2 × V100-SXM2 16 GB (NVLink) · `sm_70` · CUDA 12.8 · TP2.
A document is listed here only when its stated conditions cover that build. Everything measured or
designed for another platform is archived; the rules that decide this are in
[doc-policy.md](doc-policy.md), and the archived files are listed in
[archive/README.md](archive/README.md).

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, dual-GPU (`--tp 2`) execution, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls, dual-GPU serving, YaRN extended context, and the embedded web panel (Chat, Monitor, About) |
| [Windows](windows.md) | native Windows 11 build, portable CUDA 12.8 toolchain, stubbed media, TP2 on two V100s, and the verification gates |
| [V100 port](v100.md) | the `sm_70` numerics strategy and the Linux configure path; the "five registered artifacts" it mentions is the upstream compile surface, not this fork's |
| [Performance](performance.md) | V100 Duo measurements only; the inherited RTX 5090 campaigns, the TP2/YaRN 1M results and the per-model 5090 run records now live under [archive/inherited-5090](archive/inherited-5090/) |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

The Qwen3.6 identities are still compiled and registered, but the four `qwen3.8-27b` artifacts are
this fork's release set; all of them load through the same 27B package.

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` (third-party Swift-1.5 OracleRouter conversion, projected from NInfer v3) | [Hugging Face](https://huggingface.co/kvnxiao/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer) | Separately fine-tuned model; no model card in this repository |
| Qwen3.8-27B | `gguf-q4-k-m` (local V100 profile) | generated locally as `qwen3_8_27b_q4_k_m.ninfer` | Text/MTP only; embedded Vision objects are validation-only |
| Qwen3.8-27B | `gguf-blocks` (projected from NInfer v3) | GSQ-RCO IQ3_S v3 artifact, e.g. `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | Text/MTP only; [artifact reference §15](maintainer/qwen3.8-27b-artifact.md#15-gguf-blocks-artifact) |
| Qwen3.8-27B | `gguf-blocks-full` (third-party Coder390, projected from NInfer v3) | [Hugging Face](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer) | `Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer`; full-vocabulary MTP; Q8_0 endpoints; [artifact reference §15](maintainer/qwen3.8-27b-artifact.md#15-gguf-blocks-artifact) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

Every model card states single-`RTX 5090` / `sm_120a` requirements; that describes the official
artifact, not this fork's loader. The V100 Duo numbers are in [Performance](performance.md).

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md) — the committed results under `eval/results/` are
  inherited 5090/Linux observations; see [eval/results/README.md](../eval/results/README.md)
- `docs/notes/` — in-progress investigation notes. Not contracts: a note enters this index only
  after its conclusion is folded into a maintainer reference.

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

Runtime and Op references:

- [Small-scale concurrent inference architecture](maintainer/concurrent-inference-architecture.md) — the engine boundary reference
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)
- [Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md) — the transport design
  still applies; its `2 × RTX 5090, no NVLink, 1M` measurements and capacity conclusions do not

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.6-27B model semantics](maintainer/qwen3.6-27b-model.md)
- [Qwen3.6-27B artifact contracts, including NVFP4](maintainer/qwen3.6-27b-artifact.md)
- [Qwen3.8-27B artifact contracts, including the NVFP4 target](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)

## Planned work

These describe a target state the current source does not implement. They are not implementation
maps:

- [Softmax Attention organization and migration](maintainer/softmax-attention.md)
- [Model configuration, weight binding, and execution architecture](maintainer/model-weight-execution.md)

## Archiving

- [Applicability policy](doc-policy.md) — the five applicability values, the four decision rules,
  the disposition table and the banner format
- [Archive index](archive/README.md) — every archived file with its reason and replacement

Release notes for the current line are under [`Releases/`](Releases/): `v0.1.1-win.6` is current;
`v0.1.0-win.1` and `v0.1.1-win.1` are archived because it supersedes both.
