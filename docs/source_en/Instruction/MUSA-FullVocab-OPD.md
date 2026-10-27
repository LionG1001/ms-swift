# Full-vocabulary reverse KL on MUSA: implementation and evidence

This is an experimental personal-fork review change, not an upstream release or DeepSeek-V4-Flash-0731 certification.

## Objective

Student-generated trajectories are treated as fixed during each update. The loss differentiates the complete Student distribution: `sum_v P_s(v) * (log P_s(v) - log P_t(v))`. Teacher probabilities are frozen. This differs from teacher-top-k forward KL and sampled-token policy-gradient OPD. Pure online reverse-KL GKD uses `lmbda=1`, `beta=1`, temperature1 and `sft_alpha=0`; causal response positions and EOS/padding must align.

## File-level design

- `teacher_payload.py` defines immutable Teacher/checkpoint/head/tokenizer/template/projection identities, token/position/policy-bound requests, pre-head captures and frozen head storage.
- `PreHeadTeacherProvider` hooks the actual output-head input after final normalization, converts SBH/BSH layout explicitly, captures the weight passed to tied embeddings and requires exactly one head call per request.
- `TeacherHeadStore` checks the head bytes and independently snapshots frozen weights. Reconstruction rejects identity, trajectory, temperature, token or position mismatches.
- `gkd_loss.py` consumes the new payload while preserving existing logits/top-k branches. Hidden mode requires a complete identical vocabulary and common FP32 probability computation.
- `MegatronGKDTrainer` wires local capture/reconstruction checks. `gkd_teacher_payload_mode=logits` and `gkd_lm_head_precision=native` preserve defaults.
- The opt-in FP32 head performs differentiable casts on the original parameters. It does not replace an optimizer Parameter or break weight tying.

## Explicit limits

Hidden payloads require a separate frozen local Teacher and TP=PP=CP=1. API teachers, top-k and dynamic self-distillation are rejected. FP32 projection additionally requires an unbiased, non-sequence-parallel, non-gradient-fused TP1 `ColumnParallelLinear`.

The current head snapshot is CPU-backed and copied for projection, while original Teacher logits are retained as a reconstruction reference. This is a correctness path, not a VRAM reduction, persistent GPU cache, remote RPC implementation or automatic SGLang integration. Do not apply these TP1 options to a TP8 Flash job.

## Verification

`python -m pytest tests/train/test_teacher_payload.py -q` covers loss and Student hidden/head/bias gradients, frozen Teacher, empty/ragged masks, identities, causal position metadata, tied weights and FP32 projection (32 CPU cases). CPU references do not certify GPU models.

A separate historical raw-token/KV project entry completed native HF GKD150 on eight MUSA GPUs: Qwen2.5-1.5B/3B Instruct, full FP32, TF32off, batch32, accumulation4, maximum response1024, all151936 classes, 4800 samples and1,491,099 response tokens. Maximum generation/training logprob difference was0.00018883. GSM8K test scores were Teacher1141/1319, initial Student966 and trained Student957; no task gain was established. This is not an unmodified-upstream-CLI or full Flash result.

## Bridge dependencies and review

HF GKD does not require NVIDIA Megatron-Bridge. Swift Megatron defaults to the separate `mcore-bridge` project; NVIDIA Bridge is an explicitly selectable backend. This PR does not publish NeMo's historical NVIDIA Bridge adaptations.

Review the real capture point, tied parameter identity, response alignment, full-vocabulary normalization, Student gradients and fail-fast layout guards. Remote serving still needs partition-aware transport, head-version lifecycle and measured caching before production use.

The final snapshot passes all32 numeric/protocol CPU cases and repository-wide pre-commit. A missing platformdirs dependency was installed only in the isolated YAPF tool environment; the training platform was not changed.

Upstream [chunked GKD PR #10114](https://github.com/modelscope/ms-swift/pull/10114) addresses overlapping hidden-cache/chunked-head work. This fork is a reviewable historical experiment snapshot, not a competing upstream feature submission. Its TP/PP design is a useful follow-up reference; its CUDA results do not validate this fork on MUSA.
