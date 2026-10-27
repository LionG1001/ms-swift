# GKD

If you are new to GKD/OPD-RL, please refer to the [distillation documentation](../Instruction/Distillation.md) first.

GKD (Generalized Knowledge Distillation) is a training method that transfers knowledge from a teacher model to a student model by computing the Jensen-Shannon Divergence (JSD) loss between their output distributions.

## Feature Support

Megatron GKD currently supports the following features:

- **Training Modes**: Full parameter training and LoRA fine-tuning
- **Parallelism Strategies**: Context Parallel (CP), Pipeline Parallel (PP), Tensor Parallel (TP), and Expert Parallel (EP)
- **Model Support**: Compatible with LLMs and MLLMs in Megatron-SWIFT
- **Teacher Offload**: Supports offloading teacher model to CPU to save GPU memory
- **Online Generation**: Supports on-policy generation using vLLM for student model
- **Multi-turn Training**: Supports multi-turn GKD via `--multi_turn_scheduler`, sharing the same `MultiTurnScheduler` infrastructure as GRPO.

## Parameters

### GKD-specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--teacher_model` | str | - | Path or model ID of the teacher model<br>*Can be omitted when using `teacher_model_server` |
| `--teacher_model_server` | str | None | Teacher API URL; single URL or multi-teacher JSON. See [distillation docs](../Instruction/Distillation.md#multi-teacher-routing) |
| `--teacher_tag_key` | str | `"dataset"` | Column name for matching sample tags to teacher `tags` in multi-teacher routing |
| `--gkd_logits_topk` | int | None | Number of Top-K logits; required when using external API |
| `--gkd_teacher_payload_mode` | str | `logits` | Local teacher output format; `hidden_states` enables experimental pre-head capture and full-vocabulary reconstruction, with the limitations below |
| `--beta` | float | 0.5 | JSD divergence interpolation coefficient:<br>• 0.0: Forward KL<br>• 0.5: Symmetric JSD<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | On-Policy learning probability:<br>• 0.0: Pure Off-Policy<br>• 1.0: Pure On-Policy |
| `--temperature` | float | 0.9 | Temperature for sampling and loss computation |
| `--sft_alpha` | float | 0 | Mix in a  proportion of SFT loss; applied to non-student-generated completions |
| `--max_completion_length` | int | 512 | Maximum tokens for generation |

### Experimental hidden-state teacher outputs

`--gkd_teacher_payload_mode hidden_states` captures the final normalized hidden states at the actual LM-head input and reconstructs full-vocabulary logits using a frozen teacher-head snapshot. It requires a separate frozen local teacher, an identical complete vocabulary, and `TP=PP=CP=1`. Top-K, teacher APIs, and dynamic self-distillation are unsupported. The default `logits` path retains its existing behavior.

The payload carries teacher/checkpoint/head/vocabulary/template identities, trajectory and policy versions, temperature, response tokens, and positions. The receiver checks these before projection and loss computation. Set `--beta 1 --sft_alpha 0` for full-vocabulary Reverse KL. This option only changes teacher data handling; on-policy generation requires a separate integration. Dataset-response fallback with generation disabled is not OPD.

The implementation currently retains the full teacher forward logits as a per-batch reconstruction reference. The head snapshot is stored on CPU and copied to the compute device for projection. This is a correctness path, not an implementation of a fused head, reduced peak memory, or multi-teacher scheduling. Local CPU tests do not establish MUSA model or distributed correctness.

The hidden-state path retains the head/activation dtype for projection, then converts both complete logit tensors to FP32 for probabilities and KL. This avoids mixing FP32 student softmax with BF16 teacher softmax.

### Batch-related Parameters

Same as Megatron SFT, use the following parameters to control batch size:

| Parameter | Description |
|-----------|-------------|
| `--micro_batch_size` | Training batch size per DP group |
| `--global_batch_size` | Global batch size: `micro_batch_size × dp_size × gradient_accumulation_steps` |

## Reference

For more parameters, please refer to [Command-line Parameters](./Command-line-parameters.md)

For training scripts, please refer to [Megatron GKD Scripts](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd)

Training script using Teacher Server reference [here](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd/teacher_server.sh)

`--gkd_lm_head_precision float32` promotes the Teacher/Student final projection to FP32 in this path while retaining the original BF16 parameters, tied weights, and autograd connection. It requires an unbiased TP1 head without gradient fusion. Payload identity records projection precision, and HeadStore reconstructs with the same policy. Configure the online generation backend to use matching head precision; this option does not configure an inference server.
