# Fast LLM Inference

Collection of speculative decoding and inference acceleration techniques for large language models. Implements multiple SOTA approaches: Medusa multi-head prediction, Kangaroo adapter-based drafting, LayerSkip self-speculative decoding, prompt lookup decoding, and batched speculative verification.

## Implemented Methods

- **Speculative Decoding** (`speculative_decoding/`) - Classic draft-then-verify with configurable draft model and acceptance criteria
- **Medusa** (`medusa/`) - Multi-head parallel prediction with training pipeline for tree-structured draft generation
- **Kangaroo** (`kangaroo/`) - Lightweight adapter model for Llama3 and Gemma2 that generates draft tokens at minimal cost
- **Self-Speculative (LayerSkip)** (`self_speculative_decoding/`) - Uses early exit from deeper layers as the draft model with no additional parameters
- **Batched Speculative** (`batched_speculative_decoding/`) - Batch-level speculative decoding for throughput-optimized serving
- **Prompt Lookup** (`prompt_lookup_decoding/`) - N-gram matching from prompt context for zero-cost draft generation
- **CUDA Kernels** (`cuda_optimization/`) - Custom CUDA implementations for fused attention and verification

## Stack

Python / PyTorch / Transformers / CUDA

## Training

Medusa and Kangaroo adapters include full training pipelines with data generation scripts. See `medusa/train.py` and `kangaroo/train.py`.
