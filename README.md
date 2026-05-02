# ReQuant

A plug-and-play **post-quantization refinement** framework that starts from any existing PTQ
solution (RTN / GPTQ / GPTAQ / AWQ) and further optimises the quantized weights directly on
the discrete quantization grid. ReQuant is orthogonal to the base method — it reuses the
quantizer's scales/zero-points and only refines *integer codes* through coordinate descent
against an activation-aware reconstruction objective.

Supported quantizers (drop-in): **RTN**, **GPTQ**, **GPTAQ**, **AWQ**, with optional
**SpinQuant/QuaRot Hadamard rotation** for W4A4 activation quantization.

## TL;DR

| Method | W-quant | +Rotate | +ReQuant | +A4 | Entry script |
|--------|---------|---------|----------|-----|--------------|
| RTN    | ✅ | ✅ | ✅ (DP) | ✅ | `scripts/rtn.sh` |
| GPTQ   | ✅ | ✅ | ✅ (DP) | ✅ | `scripts/gptq.sh` |
| GPTAQ  | ✅ | ✅ | ✅ (DP) | ✅ | `scripts/gptaq.sh` |
| AWQ    | ✅ | ✅ | ✅ (DP) | ✅ | `scripts/awq.sh` |

All four entry scripts share the same CLI surface (same env variables such as `REQUANT`, `ROTATE`,
`A_BITS`, `N_SAMPLES`, …) and run data-parallel calibration across multiple GPUs via `torchrun`.

---

## Method

The preceding line of PTQ work (GPTQ, GPTAQ, AWQ) constructs quantized weights through **greedy
column-wise decisions**: once a column is quantized, it is fixed, and only later columns are used
for compensation. This procedure yields strong initial solutions, but it does not directly
refine earlier discrete choices with respect to the global reconstruction objective. To address
this, **ReQuant** iteratively optimises the quantized weights over the same discrete grid,
producing $\mathbf{W}_q^{(t)}$ after the $t$-th refinement sweep, and reduces the
activation-aware reconstruction objective through local coordinate updates.

### Row-wise decomposition

Because the Frobenius norm decomposes over output rows,

$$
\| \mathbf{W}\mathbf{X} - \mathbf{W}_q \widetilde{\mathbf{X}} \|_F^2
=
\sum_{i=1}^{d_{\mathrm{row}}}
\| \mathbf{W}_{i,:}\mathbf{X} - \mathbf{W}_{q,i,:}\widetilde{\mathbf{X}} \|_2^2 ,
$$

each output row can be refined independently while sharing the same activation statistics.
Fix one row and omit the row index. Let $\mathbf{w}$ be the full-precision row and
$\mathbf{q}$ be the current quantized row. Define

$$
\mathbf{e} := \mathbf{w} - \mathbf{q} , \quad \Delta\mathbf{X} := \widetilde{\mathbf{X}} - \mathbf{X}.
$$

The row-wise objective becomes

$$
L(\mathbf{e})
=
\| \mathbf{w}\mathbf{X} - \mathbf{q}\widetilde{\mathbf{X}} \|_2^2
=
\| \mathbf{e}\widetilde{\mathbf{X}} - \mathbf{w}\Delta\mathbf{X} \|_2^2 .
$$

With $\widetilde{\mathbf H}:=\widetilde{\mathbf X}\widetilde{\mathbf X}^\top$ and
$\mathbf B:=\Delta\mathbf X\widetilde{\mathbf X}^\top$, we maintain the score vector

$$
\mathbf{g} := \mathbf{e}\widetilde{\mathbf H} - \mathbf{w}\mathbf B .
$$

### Closed-form coordinate step

At coordinate $j$, a feasible grid step $\Delta q_j$ updates the error to
$\mathbf{e}' = \mathbf{e} - \Delta q_j\,\mathbf{u}_j$. The resulting objective change has the
closed form

$$
\Delta L(\Delta q_j)
= L(\mathbf{e}') - L(\mathbf{e})
= -2\,\Delta q_j\,(\mathbf{g})_j + (\Delta q_j)^2\,\widetilde{H}_{jj} .
$$

Let $z_j$, $s_j$, $o_j$ denote the integer code, scale, and zero point at coordinate $j$, so
$q_j = (z_j - o_j)s_j$. Any feasible grid step has the form $\Delta q_j = k\,s_j$ for some
nonzero integer $k$ that keeps $z_j + k$ inside the quantization range:

$$
\mathcal G_j := \{\, k\,s_j : k \in \mathbb Z,\; k \ne 0,\; z_{\min} \le z_j+k \le z_{\max}\,\} .
$$

### Refinement algorithm

ReQuant selects the feasible step with the smallest predicted change,
$\Delta q_j^\star = \arg\min_{\Delta q \in \mathcal G_j} \Delta L(\Delta q)$, and accepts it
only if its predicted reduction exceeds the per-coordinate average reconstruction loss:

$$
\boxed{\quad \Delta L(\Delta q_j^\star) < -\,\frac{L(\mathbf{e})}{d_{\mathrm{col}}} \quad}
$$

This threshold is necessary because $\widetilde{\mathbf H}$ and $\mathbf B$ are estimated from
a *finite* calibration set, so $\Delta L$ is only a sample approximation. Requiring each
accepted move to reduce the loss by at least the current per-coordinate average
$L(\mathbf e)/d_{\mathrm{col}}$ filters out updates whose apparent gains fall within the
estimation noise of $\widetilde{\mathbf H}$, and keeps refinement focused on moves that are
genuinely beneficial rather than overfitted to calibration samples.

When a step is accepted, $\mathbf e$ and $\mathbf g$ are refreshed **incrementally**; in
particular $\mathbf g$ admits a rank-one update

$$
\mathbf g \leftarrow \mathbf g - \Delta q_j^\star\,\widetilde{\mathbf H}_{j,:} .
$$

```
Algorithm  Post-Quantization Discrete Coordinate Refinement
Input :   row w (FP), q^(0) (initial quant row), grid info, H~, B, sweeps T
Init :   q <- q^(0);   e <- w - q;   g <- e H~ - w B
For  t = 1..T:
    For j = 1..d_col:
        dq* <- argmin over dq in G_j of  [ -2 dq g_j + dq^2 H~_jj ]
        If  -2 dq* g_j + (dq*)^2 H~_jj  <  -L(e) / d_col :
            q_j <- q_j + dq*;   e_j <- e_j - dq*
            g   <- g - dq* * H~_{j,:}            # rank-1 update, O(d_col)
Return q^(T)
```

### Analysis

**Convergence.** With fixed calibration statistics $\widetilde{\mathbf H}$, $\mathbf B$ and a
finite grid, the acceptance rule guarantees every accepted update contracts the row-wise loss
by at least a factor of $1 - 1/d_{\mathrm{col}}$:

$$
L(\mathbf e') \le \Bigl(1 - \tfrac{1}{d_{\mathrm{col}}}\Bigr) L(\mathbf e) .
$$

Iterating over $n$ acceptances gives $L \le (1-1/d_{\mathrm{col}})^n L_0$, so the row-wise loss
decays at least geometrically along the accepted updates. Because $L$ strictly decreases and
the feasible set $\mathcal Q$ is finite, no quantized row is revisited, and running sweeps
until no update is accepted terminates after finitely many acceptances. At termination, every
feasible step satisfies $\Delta L(\Delta q) \ge -L(\mathbf e)/d_{\mathrm{col}}$, i.e. a
coordinate-wise local optimum at relative tolerance $1/d_{\mathrm{col}}$.

**Efficiency.** Scoring one feasible step uses only two scalars $(\mathbf g)_j$ and
$\widetilde H_{jj}$, so scanning all candidates at coordinate $j$ costs
$O(|\mathcal G_j|) \le O(2^b{-}1)$ for $b$-bit quantization. When a step is accepted,
$\mathbf g$ is refreshed by one vector subtraction in $O(d_{\mathrm{col}})$; recomputing
$\mathbf g$ from scratch would cost $O(d_{\mathrm{col}}^2)$ and recomputing $L(\mathbf e)$
would cost $O(m\,d_{\mathrm{col}})$. Since the row-wise decomposition splits the layer
objective into independent row sub-problems that share $\widetilde{\mathbf H}$ and $\mathbf B$,
**the refinement runs in parallel over output rows without any inter-row synchronization.**

### Where the numbers come from in the codebase

- `H~` and `dXXT` (= $\mathbf B$): accumulated by forward hooks on the `quantized` block
  (`requant.py::requant_layer` for RTN/AWQ, or built online during GPTQ/GPTAQ Phase-1).
- Coordinate descent: `requant.py::requant_from_config` (vectorised per-row, operates on the
  integer codes directly via `scale` / `zero` / `maxq` of `WeightQuantizer`).
- Data-parallel reduction: `dist.all_reduce` over $\widetilde{\mathbf H}$ and $\mathbf B$
  so every rank runs the *same* deterministic refinement on identical statistics.
- Activation-quant awareness: when `ENABLE_AQ_CALIBRATION=1`, the activation quantizers are
  configured **before** weight quantization so ReQuant optimises under the true
  $\widetilde{\mathbf X}$ seen at inference time (e.g. A4).

---

## Repository layout

```
GPTQ_plus-main/
├── requant.py                      # ReQuant core: per-row discrete coordinate descent
├── process_args.py                 # unified CLI (all quant methods share the same argparse)
│
├── ptq.py                          # single-GPU PTQ entry
├── ptq_dp.py                       # torchrun entry: GPTQ / GPTAQ / AWQ data-parallel
├── ptq_gptq_dp.py                  # torchrun entry: GPTQ DP (legacy, kept for rollback)
├── ptq_rtn_dp.py                   # torchrun entry: RTN DP (+ optional ReQuant)
│
├── gptq_utils/
│   ├── main.py                     # single-GPU dispatcher (rtn / gptq / gptaq / awq)
│   ├── gptq_utils.py               # GPTQ fasterquant (single-GPU reference)
│   ├── gptaq_utils.py              # GPTAQ fasterquant (single-GPU reference)
│   ├── gptq_dp_utils.py            # GPTQ + GPTAQ + ReQuant DP implementation
│   ├── awq_dp_utils.py             # AWQ DP (aligned with MIT llm-awq) + auto_clip + ReQuant
│   ├── rtn_dp_utils.py             # RTN DP (layer-parallel) + ReQuant
│   ├── quantize_*_dp.py            # thin wrappers (checkpoint save / seed / device)
│   ├── dp_common.py                # shared DP helpers (FP-forward capture, shard scatter, ...)
│   └── ptq_dp_runner.py            # shared outer driver (rotate / actquant timing / eval)
│
├── utils/
│   ├── quant_utils.py              # WeightQuantizer / ActQuantizer / ActQuantWrapper
│   ├── rotation_utils.py           # SpinQuant Hadamard fuse + rotate
│   ├── hadamard_utils.py           # online Hadamard transform for down_proj
│   ├── eval_utils.py               # KL / PPL evaluation + lm-eval QA
│   ├── data_utils.py               # wikitext2 / ultrachat_2k / numinamath / neuralmagic loaders
│   ├── dist_utils.py               # torch.distributed helpers
│   └── model_utils.py              # ModelAnalyzer (Llama / Qwen2/3 layer discovery)
│
├── scripts/
│   ├── rtn.sh / gptq.sh / gptaq.sh / awq.sh
│   ├── sweep_gptaq_requant_eps.sh  # sweep REQUANT_MIN_GAIN_EPS
│   ├── eval_lm_tasks.sh            # lm-eval a saved .pt (auto-detects A_BITS / rotate)
│   ├── eval_fp_lm_tasks.sh         # lm-eval the FP baseline
│   ├── cache_ptq_local_datasets.sh # mirror PTQ datasets to ./datasets/<name>
│   └── cache_lm_eval_datasets.sh   # pre-cache lm-eval-harness datasets (parquet)
│
├── requirements.txt
└── README.md
```

---

## Install

```bash
conda create -n requant python=3.10 -y
conda activate requant
pip install -r requirements.txt
```

Pinned deps: `torch==2.9.1`, `transformers==4.56.2`, `accelerate==1.12.0`, `datasets==3.6.0`,
`lm-eval==0.4.4`, `sentencepiece==0.2.1`.

> Hardware: we test on 4× A800 (80 GB) for Llama-3-8B W4A4 GPTAQ + ReQuant.
> Single-GPU runs also work (`NPROC=1`), they just take longer.

---

## Data preparation

The four **calibration / PPL** datasets are mirrored to the local `./datasets/` directory (so
runs are fully offline after setup). These mirrors are **complete HF dataset repos (parquet +
`dataset_infos.json` + README)** so `datasets.load_dataset(path, config, split=slice)` works
unchanged — flat JSONL does **not** work because HF split-slicing and named configs require
the full repo layout.

```bash
# Optional: use the HF mirror for China networks
HF_ENDPOINT=https://hf-mirror.com \
  bash scripts/cache_ptq_local_datasets.sh

# or skip the larger ones
SKIP_NUMINAMATH=1 SKIP_NEURALMAGIC=1 \
  bash scripts/cache_ptq_local_datasets.sh
```

This mirrors, with sanity-check verify step:
- `wikitext` → `./datasets/wikitext`
- `HuggingFaceH4/ultrachat_200k` → `./datasets/ultrachat_2k`
- `AI-MO/NuminaMath-1.5` → `./datasets/NuminaMath-1.5`
- `neuralmagic/LLM_compression_calibration` → `./datasets/LLM_compression_calibration`

For the downstream QA tasks (lm-eval-harness: piqa, hellaswag, arc_easy/challenge, winogrande,
openbookqa, social_iqa, boolq, lambada_openai, ceval-valid × 52 subjects):

```bash
bash scripts/cache_lm_eval_datasets.sh
```

The cache script parses the installed `lm_eval` task YAMLs to derive the exact
`(dataset_path, dataset_name, split)` triples used at eval time, so it always matches what
`qa_eval` actually requests (no hard-coded guesses).

---

## Quick start

Every quantization script is called with the same positional arguments
`<model_path> <visible_GPUs> <nproc>` and is controlled by environment variables.

### RTN W4 + Rotate (classical round-to-nearest baseline)

```bash
ROTATE=1 bash scripts/rtn.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4
```

### GPTQ W4A4 + Rotate + ReQuant, 4-GPU DP

```bash
REQUANT=1 ROTATE=1 A_BITS=4 \
  bash scripts/gptq.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4
```

### GPTAQ W4A4 + Rotate + ReQuant, 4-GPU DP

```bash
REQUANT=1 ROTATE=1 A_BITS=4 \
  bash scripts/gptaq.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4
```

### AWQ W4A4 + Rotate + ReQuant, 4-GPU DP (MIT `llm-awq` aligned)

```bash
REQUANT=1 ROTATE=1 A_BITS=4 \
  bash scripts/awq.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4
```

Common switches that work on **all four** scripts:

| env var | default | meaning |
|---|---|---|
| `REQUANT` | `0` | enable ReQuant on top of the base quantizer |
| `REQUANT_SWEEPS` | `4` | number of coordinate-descent sweeps $T$ |
| `REQUANT_CANDIDATES` | `2` | neighbours per direction in $\mathcal G_j$ |
| `REQUANT_MIN_GAIN_EPS` | `0.2` | relative-loss threshold coefficient (acceptance rule, see Method) |
| `ROTATE` | `0` (`1` for gptq/gptaq/awq) | fuse LayerNorms + apply SpinQuant/QuaRot Hadamard rotation + online Hadamard on `down_proj` |
| `A_BITS` | `16` (`4` for gptq/gptaq/awq) | activation bits (16 = W-only; 4 = W4A4) |
| `A_CLIP_RATIO` | `0.9` | activation clip ratio |
| `A_ASYM` | `1` | asymmetric activation quant |
| `ENABLE_AQ_CALIBRATION` | `1` | configure activation quantizers **before** weight quant so ReQuant optimises under the true A4 activations |
| `W_ASYM` | `1` | asymmetric weight quant |
| `N_SAMPLES` | `512` | calibration sample count (GPTQ/GPTAQ Hessian, ReQuant H/dXXT) |
| `SEQ_LEN` | `2048` | calibration sequence length |
| `LM_EVAL` | `0` | run lm-eval QA tasks after quantization (off by default to keep runs short) |
| `LM_EVAL_BATCH_SIZE` | `32` | lm-eval per-device batch size |

AWQ-specific (`scripts/awq.sh`):

| env var | default | meaning |
|---|---|---|
| `AWQ_NSAMPLES` | `128` | AWQ scale-search & auto-clip calibration size (decoupled from `N_SAMPLES`; MIT default) |
| `AWQ_GRID` | `20` | scale-search grid size |
| `AWQ_MIN_ALPHA` / `AWQ_MAX_ALPHA` | `0.0` / `1.0` | half-open ratio range $[\alpha_{\min}, \alpha_{\max})$ |
| `AWQ_AUTO_CLIP` | `1` | enable MIT-style per-row auto-clip (recommended; disables `--w_clip` to avoid double-clip) |
| `AWQ_CLIP_N_GRID` | `20` | auto-clip grid |
| `AWQ_CLIP_MAX_SHRINK` | `0.5` | max shrinkage fraction |
| `AWQ_FC_FC_SCALE` | `0` | enable `up_proj -> down_proj` FC-FC scaling (off by default — not compatible with Hadamard rotation) |

RTN-specific (`scripts/rtn.sh`):

| env var | default | meaning |
|---|---|---|
| `W_CLIP` | `0` | `0` = classical pure per-channel min-max RTN; `1` = RTN + MSE clip-ratio search |

### Sweep `REQUANT_MIN_GAIN_EPS`

```bash
EPS_LIST="0.1 0.2 0.5 1 2" \
  bash scripts/sweep_gptaq_requant_eps.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4
```

Each `eps` gets its own `outputs/<model>/gptaq_eps_sweep_<date>_eps<eps>/...pt` so runs don't
overwrite each other.

---

## Output naming

Every quantization script writes its checkpoint into
`./outputs/<MODEL_NAME>/<EXP>/<method>_w4[<act_tag>]_ns<N>[<method_tag>][<rotate_tag>][<requant_tag>].pt`.

Examples:

```
outputs/Meta-Llama-3-8B/gptaq/gptaq_w4_a4_aasym_ns512_a0.25_rot_requant_s4_c2.pt
outputs/Meta-Llama-3-8B/gptq/gptq_w4_ns512_rot.pt
outputs/Meta-Llama-3-8B/awq_dp/awq_w4_a4_aasym_ns512_g20_a0.0-1.0_rot_requant_s4_c2.pt
outputs/Meta-Llama-3-8B/rtn/rtn_w4_a4_aasym_ns512_rot_requant_s4_c2.pt
```

Decoding the tags:
- `w4` = weight bits
- `a4_aasym` = `A_BITS=4`, `A_ASYM=1`
- `ns512` = `N_SAMPLES`
- `a0.25` = GPTAQ `ALPHA` (Phase-1 dXXT strength)
- `g20_a0.0-1.0` = AWQ grid / alpha range
- `rot` = SpinQuant/QuaRot rotation
- `requant_s4_c2` = ReQuant with `sweeps=4` `candidates=2`

---

## Evaluation

During quantization the scripts always log **KL / PPL** on `wikitext2`, `ultrachat_2k`,
`numinamath`. To also run lm-eval QA tasks (piqa / hellaswag / arc / winogrande / openbookqa /
social_iqa / boolq / lambada_openai / ceval-valid) on an existing checkpoint:

```bash
# A_BITS is auto-detected from the checkpoint filename (_w4_a4_aasym_ns... -> A4 asym)
bash scripts/eval_lm_tasks.sh \
    ./modelzoo/Llama3/Meta-Llama-3-8B \
    ./outputs/Meta-Llama-3-8B/gptaq/gptaq_w4_a4_aasym_ns512_a0.25_rot_requant_s4_c2.pt \
    0

# Multi-GPU dispatch
bash scripts/eval_lm_tasks.sh \
    ./modelzoo/Llama3/Meta-Llama-3-8B \
    ./outputs/.../xxx.pt \
    0,1,2,3

# Through an HTTP proxy (optional)
HTTP_PROXY_URL=http://proxy.example.com:7890 \
  bash scripts/eval_lm_tasks.sh ./modelzoo/... ./outputs/... 0
```

FP baseline QA run:

```bash
bash scripts/eval_fp_lm_tasks.sh ./modelzoo/Llama3/Meta-Llama-3-8B 0
```

---

## Acknowledgements

This repo builds on the open-source implementations of:

- [GPTQ](https://github.com/IST-DASLab/gptq) / [GPTQ+] weight quantization
- [GPTAQ](https://github.com/GPTAQ) activation-aware quantization (Phase-1 `dXXT` term)
- [MIT llm-awq](https://github.com/mit-han-lab/llm-awq) — the AWQ DP path (`awq_dp_utils.py`)
  is strictly aligned with MIT's `auto_scale.py` / `auto_clip.py`
- [QuaRot / SpinQuant](https://github.com/spcl/QuaRot) — Hadamard rotation (`rotation_utils.py`,
  `hadamard_utils.py`)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for downstream QA

ReQuant itself (the discrete coordinate-descent refinement on the quantization grid with the
$L(\mathbf e)/d_{\mathrm{col}}$ acceptance threshold, the rank-one $\mathbf g$ update, and the
per-method `requant_layer` glue) is implemented in this repository.
