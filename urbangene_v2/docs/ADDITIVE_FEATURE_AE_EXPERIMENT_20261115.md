# 已由整图编码融合方案替代

用户在2026-11-15明确选择：两张完整街景的编码相加，生成融合后的街景。

本文件最初基于“同一场景中的成分特征相加”的暂定理解起草。最终方案已经按用户确认重写，见：

[WHOLE_SCENE_ADDITIVE_AE_EXPERIMENT_20261115.md](WHOLE_SCENE_ADDITIVE_AE_EXPERIMENT_20261115.md)

最终主接口为 `C_hat = D(E(A) + E(B))`，重新训练图像编码器，不以Slot实例分解作为前提。本次仅修改实验设计。
