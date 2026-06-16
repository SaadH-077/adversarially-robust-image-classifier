# Adversarially Robust Image Classifier — Friendly·MART

Training an image classifier that stays accurate under worst-case $\ell_\infty$ adversarial
perturbations ($\varepsilon = 8/255$), using **Friendly Adversarial Training + MART**
(FAT·MART) with EMA weight averaging and SGDR cosine warm restarts on a ResNet-34.

> **Public leaderboard score: `0.642964`** &nbsp;|&nbsp; metric = $\tfrac12$·clean&nbsp;+&nbsp; $\tfrac12$·robust accuracy
> &nbsp;|&nbsp; clean ≈ **0.770**, PGD-20 ≈ **0.516** on held-out validation.

---

## 1. Problem

Given 50,000 images ($3\times32\times32$, 9 classes, used as `float`$/255\in[0,1]$, **no
normalization**), train a classifier robust to $\ell_\infty$ attacks. The architecture is
restricted to `resnet18/34/50` from `torchvision` with **only the final `fc` layer
replaced** to output 9 logits (the evaluation server reconstructs the bare architecture and
loads the `state_dict`). A submission is scored as

$$\text{score} \=\ \tfrac{1}{2}\,A_{\text{clean}} \ +\ \tfrac{1}{2}\,A_{\text{rob}}$$

and is rejected if $A_{\text{clean}} < 0.5$. The evaluation attack is hidden, so the goal is
*genuine* robustness, not robustness overfit to one attack.

## 2. Method — Friendly·MART

Because the score weights clean and robust accuracy equally (and clean accuracy transfers
directly to the held-out test set), a strong model must lift **both** axes. Our recipe does
this by pairing a misclassification-aware robust loss (MART) with a *friendly*,
early-stopped inner attack (FAT).

**Inner attack — PGD (Madry et al., 2018).** Adversarial examples are found by projected
gradient ascent on the cross-entropy inside the $\ell_\infty$ ball:

$$x^{(t+1)} = \Pi_{\|x'-x\|_\infty\le\varepsilon}\!\Big(x^{(t)} + \alpha\,\mathrm{sign}\big(\nabla_{x^{(t)}}\,\mathcal{L}_{\mathrm{CE}}(f(x^{(t)}),y)\big)\Big),\quad \varepsilon=\tfrac{8}{255},\ \alpha=\tfrac{2}{255}.$$

**Friendly early-stopping — FAT (Zhang et al., 2020).** Maximally adversarial examples make
training conservative and depress clean accuracy. FAT instead uses the *least*-adversarial
example: PGD is run per-sample but **frozen $\tau$ steps after the point is first
misclassified** ($f(x^{(t)})\neq y$); samples never misclassified within the step budget get
the full attack. A curriculum ramps the margin, $\tau=\min(2,\lfloor\text{epoch}/20\rfloor)$
— very friendly early (builds clean accuracy), slightly harder later (restores robustness).

**Robust loss — MART (Wang et al., 2020).** On those examples we minimize

$$\mathcal{L}_{\mathrm{MART}} = \underbrace{\mathrm{BCE}\big(f(x'),y\big)}_{\text{boosted CE on adv.}} \;+\; \beta\,\big(1 - p_f(y\mid x)\big)\,\underbrace{\mathrm{KL}\big(f(x)\,\|\,f(x')\big)}_{\text{robustness, up-weighted on misclassified}},\quad \beta=3,$$

where $x'$ is the friendly adversarial example, $p_f(y\mid x)$ is the clean predicted
probability of the true class, and $\mathrm{BCE}$ is cross-entropy plus a margin term that
suppresses the most-likely wrong class. The KL/softmax terms are computed in **fp32 with
logits clamped to $[-30,30]$** to avoid fp16 underflow producing `nan` (which silently makes
the AMP gradient scaler skip every step).

**Stabilisation.** We keep an exponential moving average (EMA, decay 0.999) of all weights
and BatchNorm buffers and **submit the EMA model**; the schedule uses SGDR cosine warm
restarts, which specifically improve PGD robustness.

### Final configuration

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

## 3. How to recreate the best result

1. **Open `train_robust_classifier.ipynb` in Google Colab** with a GPU runtime (T4 is
   enough; ~80 minutes for 100 epochs). This single notebook reproduces the best result.
2. **Run all cells, top to bottom.** The notebook:
   - downloads the dataset from
     `https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz`;
   - trains ResNet-34 with the FAT·MART recipe above, selecting the best checkpoint by
     validation `score` (EMA weights);
   - reports final clean / FGSM / PGD-20 accuracy and the estimated score;
   - saves the submission `state_dict` `resnet34_r34_fatmart_tau2_data_submission.pt`.
3. **Submit** with `submission.py`: set `API_KEY`, `MODEL_PATH` (the saved `.pt`), and
   `MODEL_NAME = "resnet34"`, then set `SUBMIT = True` and run `python submission.py`.

### Why FAT·MART
Standard adversarial training and TRADES plateau on a flat trade-off frontier; MART raises
robustness but sacrifices clean accuracy. **FAT** trains on the *least-adversarial* example,
which preserves clean accuracy — the axis that transfers one-to-one to the held-out test
set — while MART keeps robustness high. The combination lifts both axes at once.

## 4. Files
- **`train_robust_classifier.ipynb`** — the winning training notebook; run this to recreate the result.
- `submission.py` — leaderboard submission script (fill in your API key + model path).
- `Assignment_3_-_Robustness.pdf` — the task specification.
- `Other Experiments/` — every other approach we explored (PGD-AT, TRADES, MART, AWP,
  augmentation, and capacity variants on ResNet-18/34/50). These are *not* needed to
  recreate the best result; they document the full exploration behind it
  (see the Appendix table below for what each varied).

## 5. References
- Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD-AT).
- Zhang et al. (2019), *Theoretically Principled Trade-off between Robustness and Accuracy* (TRADES).
- Wang et al. (2020), *Improving Adversarial Robustness Requires Revisiting Misclassified Examples* (MART).
- Zhang et al. (2020), *Attacks Which Do Not Kill Training Make Adversarial Learning Stronger* (FAT).
- Wu et al. (2020), *Adversarial Weight Perturbation Helps Robust Generalization* (AWP).
- Loshchilov & Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts*.
- Izmailov et al. (2018), *Averaging Weights Leads to Wider Optima* (SWA / EMA).
- Croce & Hein (2020), *Reliable Evaluation of Adversarial Robustness* (AutoAttack).

---

## Appendix — Experiments explored

Every approach we tried, by what was varied (backbone, loss, inner attack, schedule,
regularizer). Listed for completeness; the final recipe is **E30**.

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

All experiments use $\varepsilon=8/255$, EMA(0.999), best-checkpoint selection, and
crop+flip augmentation unless the row states otherwise.
