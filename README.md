# Adversarially Robust Image Classifier — Friendly·MART

ResNet-34 trained with **Friendly Adversarial Training + MART (FAT·MART)**, EMA weight
averaging, and SGDR cosine warm restarts.

> **Public leaderboard score: `0.642964`** &nbsp;|&nbsp; metric = $\tfrac12$·clean + $\tfrac12$·robust accuracy
> &nbsp;|&nbsp; clean ≈ **0.770**, PGD-20 ≈ **0.516** on held-out validation.

## Problem

Train a `torchvision` ResNet (only the final `fc` replaced to output 9 logits) on 50,000
images ($3\times32\times32$, 9 classes, used as `float`$/255 \in [0,1]$, **no normalization**)
that is robust to a hidden $\ell_\infty$ attack at $\varepsilon = 8/255$. Submissions are
scored as $\tfrac12 A_{\text{clean}} + \tfrac12 A_{\text{rob}}$ and rejected below 50% clean
accuracy.

## Method

ResNet-34 with the **MART** loss (a misclassification-aware robust loss) and a **friendly,
early-stopped PGD inner attack (FAT)**, plus EMA and SGDR. FAT trains on the *least*-adversarial
example, which preserves clean accuracy (the axis that transfers directly to the test score),
while MART keeps robustness high, so both halves of the score rise together. The KL/softmax
terms are computed in fp32 with logits clamped to $[-30, 30]$, which prevents an fp16 underflow
that otherwise yields `nan` and stalls training.

### Configuration

| Component | Value |
|---|---|
| Backbone | `torchvision.models.resnet34(weights=None)`, `fc` → `Linear(·, 9)` |
| Loss | MART, $\beta = 3.0$ (boosted-CE + $(1-p_{\text{clean}}[y])$-weighted KL) |
| Inner attack | **FAT** early-stopped $\ell_\infty$ PGD, max 10 steps, $\varepsilon=8/255$, $\alpha=2/255$ |
| FAT curriculum | $\tau = \min(2,\ \lfloor \text{epoch}/20 \rfloor)$ |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay $5\times10^{-4}$ |
| Schedule | warmup(3) → cosine, SGDR restarts `[40, 60, 80, 100]`, 100 epochs |
| Weight averaging | EMA, decay 0.999 (all params + BN buffers); EMA weights submitted |
| Augmentation | RandomCrop(32, pad 4, reflect) + RandomHorizontalFlip |
| Precision | AMP fp16; KL in fp32 with logits clamped to $[-30, 30]$ |
| Data | 49,000 train / 1,000 held-out validation (seed 42) |
| Checkpoint | best $\tfrac12 A_{\text{clean}} + \tfrac12 A_{\text{PGD-7}}$ on the held-out split |

## How to recreate the best result

1. **Open `train_robust_classifier.ipynb` in Google Colab** with a GPU runtime (T4 is enough;
   ~80 minutes for 100 epochs). This single notebook reproduces the best result.
2. **Run all cells, top to bottom.** The notebook:
   - downloads the dataset from
     `https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz`;
   - trains ResNet-34 with the FAT·MART recipe above, selecting the best checkpoint by
     validation `score` (EMA weights);
   - reports final clean / FGSM / PGD-20 accuracy and the estimated score;
   - saves the submission `state_dict` `resnet34_r34_fatmart_tau2_data_submission.pt`.
3. **Submit** with `submission.py`: set `API_KEY`, `MODEL_PATH` (the saved `.pt`), and
   `MODEL_NAME = "resnet34"`, then set `SUBMIT = True` and run `python submission.py`.

## Files

- **`train_robust_classifier.ipynb`** — the winning training notebook; run this to recreate the result.
- `submission.py` — leaderboard submission script (fill in your API key + model path).
- `Assignment_3_-_Robustness.pdf` — the task specification.
- `Other Experiments/` — every other approach we explored (PGD-AT, TRADES, MART, AWP,
  augmentation, and capacity variants on ResNet-18/34/50). These are *not* needed to recreate
  the best result; they document the full exploration behind it (see the appendix table).


## Appendix — Experiments explored

Every approach we tried, by what was varied (backbone, loss, inner attack, schedule,
regularizer). Listed for completeness; the final recipe is **E30**. Code for each is in
`Other Experiments/`.

| # | Backbone | Loss | Inner attack | LR schedule | Key change explored |
|---|---|---|---|---|---|
| E1  | ResNet-18 | CE (PGD-AT) | PGD-7 | MultiStep | baseline (Madry) |
| E2  | ResNet-18 | CE (PGD-AT) | PGD-7 | warmup→cosine, lr 0.1 | schedule + peak-LR |
| E3  | ResNet-18 | CE (PGD-AT) | PGD-7 | cosine, lr 0.02 | lower peak LR |
| E4  | ResNet-50 | CE (PGD-AT) | PGD-7 | cosine | capacity (plain AT) |
| E5  | ResNet-18 | TRADES (β=3) | PGD-7 | cosine | TRADES loss + fp32-KL/clamp fix |
| E6  | ResNet-18 | TRADES (β=3) | PGD-7 | cosine | + EMA (0.999) |
| E7  | ResNet-18 | TRADES (β=1.5) | PGD-7 | cosine | β sweep |
| E8  | ResNet-18 | TRADES (β=3) | PGD-7 | SGDR-2 | warm restart |
| E9  | ResNet-18 | TRADES (β=3) | PGD-7 | SGDR-3 | extra restart |
| E10 | ResNet-18 | TRADES (β=3) | PGD-7 | SGDR-3 | + AWP (fixed-γ, first attempt) |
| E11 | ResNet-18 | TRADES (β=3) | PGD-7 | SGDR-3 | + label smoothing 0.1 |
| E12 | ResNet-18 | TRADES (β=3) | PGD-7 | SGDR-3 | + RandomErasing |
| E13 | ResNet-34 | TRADES (β=3) | PGD-7 | SGDR | + CutMix |
| E14 | ResNet-18 | TRADES (β=4) | PGD-7 | SGDR-3 | β sweep (up) |
| E15 | ResNet-18 | TRADES (β=2.5) | PGD-7 | SGDR-3 | β sweep |
| E16 | ResNet-18 | TRADES (β=3) | PGD-10 | SGDR-3 | stronger inner attack |
| E17 | ResNet-34 | TRADES (β=3) | PGD-7 | smooth cosine | capacity |
| E18 | ResNet-34 | TRADES (β=3) | PGD-7 | SGDR-3 | capacity + restarts |
| E19 | ResNet-34 | MART (β=5) | PGD-7 | SGDR | MART loss |
| E20 | ResNet-34 | MART (β=2) | PGD-7 | SGDR | MART β sweep |
| E21 | ResNet-34 | MART (β=3) | PGD-7 | SGDR (120 ep) | MART β=3 |
| E22 | ResNet-34 | MART (β=3) | PGD-7 | SGDR (180 ep) | longer (clean push) |
| E23 | ResNet-18 | MART (β=3) | PGD-7 | SGDR `[20,35,50]` | short schedule |
| E24 | ResNet-18 | **FAT·MART** (β=3) | **FAT** PGD, τ 0→4 | SGDR `[40,60,80,100]` | **FAT introduced** |
| E25 | ResNet-18 | FAT·MART (β=3) | FAT PGD, τ 0→3 | SGDR | + AWP (proxy form) |
| E26 | ResNet-34 | FAT·MART (β=3) | FAT PGD, τ cap 3 | SGDR `[40,60,80,100]` | capacity |
| E27 | ResNet-34 | FAT·MART (β=3) | FAT PGD, τ cap 3 | SGDR | + AWP (proxy form) |
| E28 | ResNet-34 | FAT·MART (β=3) | FAT PGD, τ cap 3 | SGDR `[40,60,80,100,140]` | longer schedule |
| E29 | ResNet-34 | FAT·MART (β=3) | FAT PGD, **τ cap 2** | SGDR `[40,60,80,100]` | friendlier curriculum |
| **E30** | **ResNet-34** | **FAT·MART (β=3)** | **FAT PGD, τ cap 2** | **SGDR `[40,60,80,100]`** | **+ 49k data → final** |

All experiments use $\varepsilon=8/255$, EMA(0.999), best-checkpoint selection, and crop+flip
augmentation unless the row states otherwise.
