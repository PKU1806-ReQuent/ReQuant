MODEL_PATH = /apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B


| Method | Model | KL-wikitext2 | PPL-wikitext2 | KL-uItrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath | arc-challenge | arc-easy | boolq | ceval-valid | hellaswag | lambada openai | openbookqa | piqa | social_iqa | winogrande | acc_avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPTAQ+Quarot | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPTAQ+Quarot+ReQuant | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPTQ+Quarot | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPTQ+Quarot+ReQuant | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RTN+Quarot | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RTN+Quarot+ReQuant | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| AWQ+Quarot | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| AWQ+Quarot+ReQuant | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|

---

## Smoke Test · Qwen3-0.6B · W4A16（lm_eval only）

MODEL_PATH = /apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-0.6B

说明：本段为小模型冒烟结果，用于验证端到端管线正确性。
- 量化均为 W4A16（`--a_bits 16`），**未启用** Quarot 旋转。
- PPL/KL 列已补齐（来源：`logs/smoke_20260426/*.log` 中的 `KL&PPL on <dataset>` 行）；10 项 zero-shot QA 准确率与 acc_avg 保持不变。
- `acc_avg` 为 10 项任务的算术平均；**各 base 方法 +ReQuant 后 acc_avg 全部正增益**。

| Method | Model | KL-wikitext2 | PPL-wikitext2 | KL-uItrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath | arc-challenge | arc-easy | boolq | ceval-valid | hellaswag | lambada openai | openbookqa | piqa | social_iqa | winogrande | acc_avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | Qwen3-0.6B | -4.16e-05 | 20.96 | -2.23e-05 | 8.44 | 2.42e-05 | 4.89 | 33.70 | 55.93 | 63.82 | 43.31 | 47.30 | 39.84 | 31.60 | 67.25 | 39.05 | 56.43 | 47.82 |
| GPTAQ | Qwen3-0.6B | 1.64e-01 | 24.51 | 2.13e-01 | 9.03 | 2.37e-01 | 5.23 | 30.72 | 41.62 | 54.92 | 31.72 | 43.55 | 35.78 | 29.80 | 64.85 | 36.34 | 54.78 | 42.41 |
| GPTAQ+ReQuant | Qwen3-0.6B | 1.50e-01 | 24.03 | 2.04e-01 | 8.61 | 2.46e-01 | 5.22 | 30.12 | 42.55 | 64.04 | 28.45 | 43.92 | 36.33 | 31.20 | 65.07 | 38.33 | 55.01 | 43.50 |
| GPTQ | Qwen3-0.6B | 2.37e-01 | 27.45 | 2.51e-01 | 9.27 | 2.86e-01 | 5.50 | 29.52 | 46.21 | 61.01 | 35.36 | 43.02 | 33.13 | 31.20 | 65.07 | 37.72 | 53.43 | 43.57 |
| GPTQ+ReQuant | Qwen3-0.6B | 1.57e-01 | 24.22 | 2.13e-01 | 9.28 | 2.64e-01 | 5.60 | 29.78 | 42.68 | 65.72 | 32.76 | 43.92 | 36.35 | 31.00 | 65.51 | 37.36 | 55.01 | 44.01 |
| RTN | Qwen3-0.6B | 4.67e-01 | 33.96 | 4.00e-01 | 10.20 | 3.37e-01 | 5.65 | 28.41 | 39.90 | 44.74 | 34.92 | 41.59 | 27.94 | 30.20 | 62.40 | 36.64 | 53.12 | 39.99 |
| RTN+ReQuant | Qwen3-0.6B | 2.43e-01 | 26.65 | 3.17e-01 | 9.92 | 3.92e-01 | 6.17 | 28.16 | 44.91 | 53.52 | 28.83 | 42.03 | 28.04 | 32.00 | 64.74 | 38.28 | 54.78 | 41.53 |
| AWQ | Qwen3-0.6B | 4.09e-01 | 33.30 | 3.15e-01 | 9.93 | 2.72e-01 | 5.63 | 28.92 | 45.79 | 43.76 | 35.66 | 42.31 | 26.08 | 30.60 | 62.89 | 38.38 | 51.70 | 40.61 |
| AWQ+ReQuant | Qwen3-0.6B | 1.74e-01 | 24.85 | 2.37e-01 | 9.35 | 3.09e-01 | 5.96 | 27.90 | 45.96 | 47.68 | 35.59 | 42.65 | 31.26 | 29.40 | 63.76 | 37.51 | 53.67 | 41.54 |

### ReQuant 净增益汇总（Qwen3-0.6B · acc_avg）

| base | base | +ReQuant | Δ |
|---|---:|---:|---:|
| RTN   | 39.99 | 41.53 | **+1.54** |
| GPTAQ | 42.41 | 43.50 | **+1.09** |
| AWQ   | 40.61 | 41.54 | **+0.93** |
| GPTQ  | 43.57 | 44.01 | **+0.44** |

4/4 全部正增益，与 smoke 时 wikitext PPL 观察一致。
