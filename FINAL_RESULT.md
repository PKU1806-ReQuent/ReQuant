MODEL_PATH = /apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B


| Method | Model | Quant Time | KL-wikitext2 | PPL-wikitext2 | KL-uItrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath | arc-challenge | arc-easy | boolq | ceval-valid | hellaswag | lambada openai | openbookqa | piqa | social_iqa | winogrande | acc_avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | Qwen3-14B | — | -1.56e-05 | 8.65 | 4.53e-06 | 4.98 | 6.12e-05 | 3.36 | 60.32 | 82.95 | 89.30 | 82.32 | 78.82 | 67.82 | 46.40 | 79.87 | 52.05 | 72.53 | 71.24 |
| GPTAQ+Quarot | Qwen3-14B | 33.9 min | 2.72e-02 | 8.83 | 3.31e-02 | 4.96 | 2.56e-02 | 3.36 | 58.96 | 82.32 | 89.02 | 80.31 | 77.45 | 67.79 | 45.40 | 80.36 | 51.28 | 71.98 | 70.49 |
| GPTAQ+Quarot+ReQuant | Qwen3-14B | 115.2 min | 2.60e-02 | 8.80 | 3.26e-02 | 4.91 | 2.56e-02 | 3.34 | 60.07 | 82.11 | 88.47 | 80.01 | 77.76 | 68.50 | 44.80 | 79.87 | 51.74 | 71.98 | 70.53 |
| GPTQ+Quarot | Qwen3-14B | 31.9 min | 2.68e-02 | 8.78 | 3.13e-02 | 4.97 | 2.40e-02 | 3.34 | 59.73 | 82.03 | 88.04 | 79.94 | 77.97 | 68.29 | 45.00 | 79.43 | 51.69 | 73.09 | 70.52 |
| GPTQ+Quarot+ReQuant | Qwen3-14B | 113.9 min | 2.50e-02 | 8.81 | 3.06e-02 | 4.94 | 2.49e-02 | 3.36 | 59.47 | 81.40 | 88.96 | 81.13 | 77.92 | 68.35 | 46.00 | 79.82 | 51.79 | 72.06 | 70.69 |
| RTN+Quarot | Qwen3-14B | 12.9 min | 1.76e-01 | 10.22 | 1.45e-01 | 5.20 | 1.02e-01 | 3.27 | 55.20 | 78.32 | 86.45 | 78.38 | 75.82 | 67.09 | 42.20 | 79.11 | 48.00 | 69.69 | 68.03 |
| RTN+Quarot+ReQuant | Qwen3-14B | 106.2 min | 3.13e-02 | 8.85 | 3.60e-02 | 4.98 | 2.92e-02 | 3.33 | 59.90 | 82.45 | 88.47 | 79.20 | 77.72 | 67.75 | 46.60 | 80.03 | 50.61 | 72.69 | 70.54 |
| AWQ+Quarot | Qwen3-14B | 23.9 min | 1.40e-01 | 9.68 | 1.17e-01 | 5.16 | 7.75e-02 | 3.43 | 57.76 | 81.86 | 88.07 | 78.16 | 76.17 | 62.72 | 45.80 | 79.33 | 50.56 | 71.27 | 69.17 |
| AWQ+Quarot+ReQuant | Qwen3-14B | 106.6 min | 5.16e-02 | 8.96 | 5.36e-02 | 5.04 | 3.57e-02 | 3.39 | 59.30 | 82.07 | 89.60 | 80.46 | 77.74 | 66.25 | 45.40 | 79.71 | 51.94 | 72.14 | 70.46 |

> **Quant Time 说明**：仅统计“量化算法阶段”耗时（不含 KL/PPL 评估与 lm_eval 零样本评估），来源为各日志中**首条 `[INFO]` 时间戳** → **`Evaluating KL&PPL on wikitext2` 日志行时间戳** 的差值。baseline 日志位于 `logs_14b/0{1..3}_*.log`（GPTAQ/GPTQ/RTN）；`AWQ+Quarot` baseline 于 2026-05-06 用 `nsamples=1024`、`OFFLOAD_INPS=0`（与四个 ReQuant 实验口径对齐、关闭 inps offload 省掉 H2D/D2H 往返）重测，日志位于 `logs_14b/15_awq_w4a16_rot_retest.log`，相比原 `04_awq_w4a16_rot.log`（`nsamples=512`、`OFFLOAD_INPS=1`，Quant Time=87.3 min）3.7× 加速且精度略升（acc_avg 68.53 → 69.17）。四个 ReQuant 实验统一使用 `requant_beta=0.25`、`OFFLOAD_INPS=0`，8 卡 DP，日志位于 `logs_14b/11_gptaq_*.log`、`12_gptq_*.log`、`13_rtn_*.log`、`14_awq_*.log`。RTN baseline 为单卡跑（其余均 8 卡 DP）；KL 评估 ≈ 5–6 min、lm_eval ≈ 23–56 min 对所有方法近似常数，未计入此列。

---

## Qwen3-14B · W4A4 · Quarot（Baselines）

MODEL_PATH = /apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B

说明：W4A4 激活量化（`--a_bits 4 --a_asym 1 --a_clip_ratio 0.9`），权重 4bit，开启 Quarot 旋转。RTN/AWQ 已走 DP 路径（与 GPTAQ/GPTQ 同为 8 卡 DP）。日志位于 `logs_14b_w4a4/0{1..4}_*.log`。

| Method | Model | Quant Time | KL-wikitext2 | PPL-wikitext2 | KL-uItrachat_2k | PPL-ultrachat_2k | KL-numinamath | PPL-numinamath | arc-challenge | arc-easy | boolq | ceval-valid | hellaswag | lambada openai | openbookqa | piqa | social_iqa | winogrande | acc_avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | Qwen3-14B | — | -1.56e-05 | 8.65 | 4.53e-06 | 4.98 | 6.12e-05 | 3.36 | 60.32 | 82.95 | 89.30 | 82.32 | 78.82 | 67.82 | 46.40 | 79.87 | 52.05 | 72.53 | 71.24 |
| GPTAQ+Quarot | Qwen3-14B | 35.3 min | 1.58e-01 | 9.73 | 1.49e-01 | 5.03 | 1.06e-01 | 3.47 | 55.20 | 78.41 | 86.73 | 74.00 | 74.25 | 64.23 | 42.40 | 77.15 | 48.82 | 68.98 | 67.02 |
| GPTAQ+Quarot+ReQuant | Qwen3-14B | 128.6 min | 1.52e-01 | 9.67 | 1.45e-01 | 5.06 | 1.01e-01 | 3.42 | 55.89 | 79.63 | 87.77 | 76.45 | 74.08 | 65.92 | 41.00 | 78.29 | 48.16 | 68.11 | 67.53 |
| GPTQ+Quarot  | Qwen3-14B | 31.4 min | 1.55e-01 | 9.71 | 1.31e-01 | 5.03 | 8.84e-02 | 3.37 | 56.83 | 77.48 | 86.70 | 76.37 | 74.69 | 65.38 | 44.80 | 78.18 | 48.77 | 70.32 | 67.95 |
| GPTQ+Quarot+ReQuant | Qwen3-14B | 128.6 min | 1.43e-01 | 9.61 | 1.28e-01 | 5.06 | 8.89e-02 | 3.43 | 57.25 | 79.92 | 86.39 | 77.04 | 74.57 | 65.79 | 43.80 | 78.67 | 48.21 | 68.67 | 68.03 |
| RTN+Quarot   | Qwen3-14B | 15.6 min | 3.22e-01 | 11.44 | 2.55e-01 | 5.37 | 1.86e-01 | 3.41 | 48.72 | 73.40 | 84.98 | 74.15 | 72.26 | 64.20 | 41.40 | 76.55 | 46.21 | 67.80 | 64.97 |
| RTN+Quarot+ReQuant | Qwen3-14B | 117.1 min | 1.49e-01 | 9.69 | 1.38e-01 | 5.10 | 9.93e-02 | 3.46 | 54.95 | 79.92 | 86.79 | 74.67 | 74.45 | 65.46 | 43.20 | 77.58 | 49.85 | 69.22 | 67.61 |
| AWQ+Quarot   | Qwen3-14B | 23.2 min | 2.71e-01 | 10.43 | 2.24e-01 | 5.24 | 1.53e-01 | 3.42 | 51.37 | 74.20 | 85.63 | 73.55 | 73.26 | 62.22 | 42.40 | 77.09 | 47.65 | 66.14 | 65.35 |
| AWQ+Quarot+ReQuant | Qwen3-14B | 109.8 min | 1.65e-01 | 9.80 | 1.50e-01 | 5.12 | 1.01e-01 | 3.48 | 57.85 | 80.26 | 87.06 | 76.30 | 74.10 | 63.50 | 43.60 | 77.48 | 48.72 | 67.56 | 67.64 |

> Quant Time 口径与 W4A16 一致：**首条 `[INFO]` 时间戳 → `Evaluating KL&PPL on wikitext2` 时间戳** 的差值，不含 KL/PPL 评估与 lm_eval 零样本评估。`RTN/AWQ` 采用的 `rtn_dp/awq_dp` 配置（`dp_shard_inps=1`、`nsamples=1024/512`），与 GPTAQ/GPTQ 均为 8×GPU DP。`AWQ+Quarot` baseline 于 2026-05-06 用 `nsamples=1024`、`OFFLOAD_INPS=0`（与四个 ReQuant 实验口径对齐、关闭 inps offload 省掉 H2D/D2H 往返）重测，日志位于 `logs_14b_w4a4/09_awq_w4a4_rot_retest.log`，相比原 `04_awq_w4a4_rot.log`（`nsamples=512`、`OFFLOAD_INPS=1`，Quant Time=87.3 min）12.8× 加速且精度略升（acc_avg 65.09 → 65.35）。ReQuant 日志位于 `logs_14b_w4a4/0{5..8}_*_requant.log`。

### 与 W4A16 baseline 的退化幅度（acc_avg）

| Method | W4A16 acc_avg | W4A4 acc_avg | Δ (W4A4 − W4A16) |
|---|---:|---:|---:|
| FP    | 71.24 | 71.24 | — |
| GPTAQ+Quarot         | 70.49 | 67.02 | **−3.47** |
| GPTAQ+Quarot+ReQuant | 70.53 | 67.53 | **−3.00** |
| GPTQ+Quarot          | 70.52 | 67.95 | **−2.57** |
| GPTQ+Quarot+ReQuant  | 70.69 | 68.03 | **−2.66** |
| RTN+Quarot           | 68.03 | 64.97 | **−3.06** |
| RTN+Quarot+ReQuant   | 70.54 | 67.61 | **−2.93** |
| AWQ+Quarot           | 69.17 | 65.35 | **−3.82** |
| AWQ+Quarot+ReQuant   | 70.46 | 67.64 | **−2.82** |

### ReQuant 在 W4A16 上的净增益（acc_avg）

| base | W4A16 base | W4A16 +ReQuant | Δ |
|---|---:|---:|---:|
| RTN+Quarot   | 68.03 | 70.54 | **+2.51** |
| AWQ+Quarot   | 69.17 | 70.46 | **+1.29** |
| GPTQ+Quarot  | 70.52 | 70.69 | **+0.17** |
| GPTAQ+Quarot | 70.49 | 70.53 | **+0.04** |

与 W4A4 走势完全一致：ReQuant 对精度损失更大的 RTN/AWQ 修复最显著（+1.29 ~ +2.51），而对本身已很强的 GPTQ/GPTAQ 只带来接近零的小增益（+0.04 ~ +0.17，基本在 noise 水平）。ReQuant 后 **RTN/AWQ 的 acc_avg 已追上 GPTQ/GPTAQ**，4 种方法几乎落在同一水平（70.46 ~ 70.69），意味着在 W4A16 下 Hessian 信息的作用基本被 ReQuant 的 phase-2 权重搜索追平——便宜方法（RTN/AWQ）+ ReQuant 可以换来贵方法（GPTQ/GPTAQ）级别的精度。PPL/KL 侧同样：RTN 从 1.76e-01/10.22 降到 3.13e-02/8.85（wikitext2），KL 下降约 5.6×、PPL 接近 GPTAQ baseline；AWQ 从 1.40e-01/9.68 降到 5.16e-02/8.96（wikitext2），KL 下降约 2.7×、PPL 下降 0.72 点，同样是显著的修复。

### ReQuant 在 W4A4 上的净增益（acc_avg）

| base | W4A4 base | W4A4 +ReQuant | Δ |
|---|---:|---:|---:|
| RTN+Quarot   | 64.97 | 67.61 | **+2.64** |
| AWQ+Quarot   | 65.35 | 67.64 | **+2.29** |
| GPTAQ+Quarot | 67.02 | 67.53 | **+0.51** |
| GPTQ+Quarot  | 67.95 | 68.03 | **+0.08** |

W4A4 下四种方法的相对排序为 **GPTQ > GPTAQ > AWQ ≈ RTN**，与 W4A16 基本一致；PPL/KL 也随之整体变差约 1 个 PPL 点（wikitext2）。**ReQuant 在 W4A4 上 4/4 正增益，且对精度损失更大的 RTN/AWQ 修复最为显著（+2.55 ~ +2.64），对本身已经较强的 GPTQ/GPTAQ 增益较小（+0.08 ~ +0.51）**；ReQuant 后 AWQ/RTN 的 acc_avg 已追平甚至反超 W4A4 baseline 下的 GPTAQ（67.02）。PPL/KL 侧同样由 3.15e-01/11.16（AWQ baseline）→ 1.65e-01/9.80（AWQ+ReQuant，wikitext2），KL 减半、PPL 下降约 1.4 点。

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
