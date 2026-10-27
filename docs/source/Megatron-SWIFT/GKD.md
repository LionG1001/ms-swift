# GKD

如果你是首次使用 GKD/OPD-RL，请先参考 [蒸馏文档](../Instruction/Distillation.md)。

GKD（Generalized Knowledge Distillation，广义知识蒸馏）是一种将教师模型的知识迁移到学生模型的训练方法，通过计算两个模型输出分布之间的 Jensen-Shannon 散度（JSD）损失来实现知识蒸馏。

## 功能支持

Megatron GKD 当前已支持以下功能：

- **训练模式**：全参数训练与 LoRA 微调
- **并行策略**：支持上下文并行（CP）、流水线并行（PP）、张量并行（TP）和专家并行（EP）
- **模型支持**：兼容 Megatron-SWIFT 中的 LLM 及 MLLM
- **Teacher Offload**：支持将教师模型卸载到 CPU 以节省 GPU 显存
- **在线生成**：支持使用 vLLM 进行学生模型的 on-policy 生成

## 参数说明

### GKD 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teacher_model` | str | - | 教师模型路径或模型 ID<br>*使用 `teacher_model_server` 时可省略 |
| `--teacher_model_server` | str | None | 教师 API 地址；支持单 URL 或多 teacher JSON，详见[蒸馏文档](../Instruction/Distillation.md#multi-teacher多教师路由) |
| `--teacher_tag_key` | str | `"dataset"` | 多 teacher 路由时样本匹配 teacher `tags` 的字段名 |
| `--gkd_logits_topk` | int | None | Top-K logits 数量，使用外部教师 API 时必须设置 |
| `--gkd_teacher_payload_mode` | str | `logits` | 本地教师输出格式；`hidden_states` 为实验性 pre-head 捕获及全词表重建路径，限制见下文 |
| `--beta` | float | 0.5 | JSD 散度插值系数：<br>• 0.0: Forward KL<br>• 0.5: 对称 JSD<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | On-Policy 学习触发概率：<br>• 0.0: 纯 Off-Policy<br>• 1.0: 纯 On-Policy |
| `--temperature` | float | 0.9 | 温度参数，用于采样和损失计算 |
| `--sft_alpha` | float | 0 | 混合一定比例的sft loss，对非student生成结果生效 |
| `--max_completion_length` | int | 512 | 生成时的最大 token 数 |

### 实验性 hidden-state 教师输出

`--gkd_teacher_payload_mode hidden_states` 从实际 LM head 的输入处捕获最终归一化后的隐藏状态，并用冻结教师的 head 快照重建完整词表 logits。它要求独立的本地冻结教师、相同的完整词表及 `TP=PP=CP=1`，不支持 Top-K、教师 API 或动态自蒸馏。默认的 `logits` 路径保持原有行为。

数据包括教师／checkpoint／head／词表／模板身份、轨迹、策略版本、温度、响应 token 与位置；接收端在投影及计算损失前检查这些信息。设置 `--beta 1 --sft_alpha 0` 使用全词表 Reverse KL。该参数只改变教师数据路径；在线生成仍需另外接入，不能把关闭生成后使用数据集响应的行为视为 OPD。

当前实现保留完整教师前向 logits 作为逐批重建参考，head 快照在 CPU 保存并在投影时复制到计算设备。它用于正确性验证，不代表 fused head、低显存峰值或多教师调度已经实现；本地 CPU 回归不代替 MUSA 实际模型及分布式验收。

hidden路径保留原head／隐藏状态的投影dtype，在投影后将两侧完整logits转为FP32计算概率及KL，避免Student FP32／Teacher BF16混合softmax精度。

### 批量相关参数

与 Megatron SFT 相同，使用以下参数控制批量大小：

| 参数 | 说明 |
|------|------|
| `--micro_batch_size` | 每个DP组的训练批次大小 |
| `--global_batch_size` | 全局批次大小：`micro_batch_size × dp_size × gradient_accumulation_steps` |

## 参考

更多参数请参考[命令行文档](./Command-line-parameters.md)

训练脚本请参考 [Megatron GKD 脚本](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd)

使用 Teacher Server 的训练脚本请参考 [这里](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd/teacher_server.sh)

`--gkd_lm_head_precision float32` 可在此路径中将 Teacher／Student 最后投影显式提升到 FP32，保持原 BF16 参数、tied weight 和梯度连接；仅支持无 bias、无梯度融合的 TP1 head。Payload 身份记录投影精度，HeadStore 按相同精度重建。在线生成后端也须显式设置相同 head 精度；该选项不会自动配置推理服务。
