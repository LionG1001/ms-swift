# MUSA上的全词表Reverse KL：实现与验证说明

本文对应个人fork的实验适配，目标是固定源码并供人工review。它不是上游正式发布，也不意味着DeepSeek-V4-Flash-0731已经跑通。当前验证基线为原生HF GKD的小模型FP32流程；Megatron hidden-state路径的支持范围另列。

## 1. 损失与在线生成是两个独立选择

Student先按当前权重生成响应。给定题目和已生成前缀，Teacher给出下一token分布。`--lmbda 1 --beta 1 --temperature 1 --sft_alpha 0`分别选择纯在线学生轨迹、Reverse KL、共同温度及不混合SFT。完整目标为：

`KL(P_student || P_teacher) = sum_v P_student(v) * (log P_student(v) - log P_teacher(v))`。

v覆盖完整输出头，不是Teacher top-k，也不是只取采样token的logprob比。响应位置的mask、EOS与因果偏移必须对齐。生成轨迹在更新时视为固定数据，梯度通过Student logits传播，Teacher不接收梯度；这不是整句采样分布的完整策略梯度。

## 2. 本次代码变化

| 文件／对象 | 改变及原因 |
| --- | --- |
| `swift/rlhf_trainers/teacher_payload.py` | 定义Teacher身份、请求、pre-head payload与冻结head存储；避免将不匹配的hidden／head／轨迹合成教学分布 |
| `TeacherIdentity` | 绑定Teacher、checkpoint、head指纹、tokenizer、模板及投影dtype；不是只信任模型名字 |
| `PreHeadTeacherProvider` | 在真实LM head输入处捕获最终归一化后的hidden；支持BSH／SBH布局，捕获tied head实际传入的weight，要求每次恰好一次head调用 |
| `TeacherHeadStore` | 校验head真实字节并保存独立冻结快照；投影前核对token、位置、policy与温度，拒绝过期／错位数据 |
| `gkd_loss.py / TeacherOutput` | 增加hidden payload消费；保留已有logits／top-k路径，hidden模式要求完整相同词表，排除混合精度softmax |
| `MegatronGKDTrainer` | 接入本地冻结Teacher捕获与完整logits重建对照；增加显式范围检查，避免把不支持的并行布局当作可用 |
| `gkd_teacher_payload_mode` | 默认`logits`保持原行为；`hidden_states`显式启用实验路径 |
| `gkd_lm_head_precision` | 默认`native`；`float32`在原参数上做可微FP32投影，不替换optimizer持有的Parameter或破坏tied weight |

`_enable_fp32_gkd_head`仅允许TP1的无bias、无sequence parallel、无gradient accumulation fusion的ColumnParallelLinear。输入和真实weight转FP32后调用`F.linear`，梯度沿cast回到原参数。Teacher使用相同投影精度。

## 3. 支持边界和成本

- hidden-state模式目前仅支持独立本地冻结Teacher及TP=PP=CP=1；拒绝Teacher API、top-k、动态自蒸馏和未支持并行布局。
- 捕获的是实际head输入；随意选择中间层输出不能精确重建Teacher完整分布。
- head快照当前驻留CPU，在投影时拷到计算设备。它是正确性实现，不是高性能常驻缓存。
- 当前保留原Teacher完整logits做重建对照，因此不宣称降低了完整前向的显存峰值。
- 这不是已完成的远程Teacher RPC，也没有自动接入SGLang或配置推理端FP32 head。
- Flash多卡可以评估原生Megatron完整logits路径；不能将本实验TP1 hidden／FP32 head参数直接套给TP8。

## 4. 测试与历史GPU证据

```bash
python -m pytest tests/train/test_teacher_payload.py -q
```

32项CPU数值／协议回归覆盖完整词表loss、Student hidden／head／bias梯度、Teacher无梯度、ragged／空mask、身份／token／位置／温度错误、tied weight及FP32投影。CPU参考不是GPU模型验收。

历史实际MUSA工程验证另有原生HF GKD的150次更新：Qwen2.5-1.5B-Instruct为Student、3B-Instruct为冻结Teacher，单机8卡、全FP32／TF32关闭、GBS32、micro1／累积4、响应上限1024。完整151936类，4800独立样本、1,491,099响应token，4800概率批次通过，最大采样／训练logprob差0.00018883。

固定GSM8K test1319题：Teacher1141、Student起点966、Student150后957。没有证明任务净收益。这个结果来自带原始token／KV与证据回调的项目入口，不能冒称上游CLI不加适配即可得到相同结果，更不能认证本PR所有后端或Flash本体。

## 5. 桥接依赖

HF/GKD不需要NVIDIA Megatron-Bridge。Swift Megatron默认使用`mcore-bridge`／`mcore_bridge`，它是不同项目；也支持显式选择NVIDIA`bridge_backend=megatron-bridge`。本轮没有整理NeMo所用的NVIDIA Bridge工作树。Flash后续优先审计实际mcore-bridge／MUSA模型适配，而不是根据库名推断兼容。

## 6. Review重点

核对捕获点和tied参数身份、响应mask、完整词表归一化及Student梯度；检查每个不支持布局是否明确报错。后续Teacher服务化需要实现并行分片、head版本生命周期、传输和缓存，不应将当前CPU快照设计直接称为生产优化。

本轮在固定基线上完成32项CPU回归及全仓库`pre-commit run --all-files`。YAPF工具环境缺少platformdirs时，仅补齐隔离的lint工具依赖；没有修改训练镜像依赖。

上游已有[chunked GKD PR #10114](https://github.com/modelscope/ms-swift/pull/10114)，包含hidden缓存与分块head/loss设计。本fork保存已做实验的独立适配供review，不作为重复的上游功能提案；后续优先比较其TP／PP设计和MUSA kernel可移植性，不能把该PR的CUDA结果当成本forkGPU验证。
