# Adversarially Robust Image Classifier (Friendly–MART)

Training an image classifier that stays accurate under worst-case L∞ adversarial
perturbations (ε = 8/255), using **Friendly Adversarial Training + MART** with EMA weight
averaging and SGDR warm restarts.

**Public leaderboard score: 0.627676** (metric: 0.5·clean + 0.5·robust accuracy).

The model is a **ResNet-34** trained with **FAT-MART** (Friendly Adversarial Training +
MART) plus EMA weight averaging and SGDR cosine warm restarts. It reaches
**clean ≈ 0.753 / PGD-20 ≈ 0.515** on a held-out validation split.

| Component | Value |
|---|---|
| Backbone | `torchvision.models.resnet34(weights=None)`, final `fc` → `Linear(·, 9)` |
| Loss | MART (boosted-CE on adv + (1 − p_clean[y])-weighted KL(clean‖adv)), β = 3.0 |
| Inner attack | **FAT** early-stopped L∞ PGD, max 10 steps, ε = 8/255, α = 2/255 |
| FAT curriculum | τ = min(3, epoch // 20) (friendly early → harder late) |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 5e-4 |
| Schedule | warmup(3) → cosine with SGDR restarts `[40, 60, 80, 100]`, 100 epochs |
| Weight averaging | EMA, decay 0.999 (all params + BN buffers); EMA weights are submitted |
| Augmentation | RandomCrop(32, pad 4, reflect) + RandomHorizontalFlip |
| Precision | AMP fp16; KL computed in fp32 with logits clamped to [−30, 30] |
| Checkpoint | best `val_score = 0.5·clean + 0.5·PGD-7` on a 2k held-out split |

## How to recreate the best leaderboard result

1. **Open `train_resnet34_fatmart.ipynb` in Google Colab** with a GPU runtime
   (T4 is sufficient; ~80 min for 100 epochs).
2. **Run all cells.** The notebook:
   - downloads the dataset automatically from
     `https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz`
     (50,000 images, 3×32×32, 9 classes, used as `float/255` in [0,1], no normalization);
   - trains ResNet-34 with the FAT-MART recipe above, selecting the best checkpoint by
     `val_score` on a 2,000-image held-out split (seed 42);
   - reports final clean / FGSM / PGD-20 accuracy and the estimated score;
   - saves the submission state_dict `resnet34_r34_fatmart_submission.pt`.
3. **Submit** with `submission.py`: set `API_KEY`, `MODEL_PATH` (the saved `.pt`), and
   `MODEL_NAME = "resnet34"`, then set `SUBMIT = True` and run `python submission.py`.
   (One submission per group per 60 minutes.)

### Why FAT-MART
Plain adversarial training and TRADES plateaued on a flat clean+robust ≈ 1.19 frontier;
MART raised robustness but sacrificed clean accuracy. **FAT** trains on the
*least-adversarial* example (PGD is stopped per-sample once it is misclassified, with a
curriculum margin τ), which preserves clean accuracy — the axis that transfers directly to
the held-out server — while MART keeps robustness high. The combination lifts both axes
(clean ≈ 0.75, PGD-20 ≈ 0.51), breaking the frontier to score 0.6277.

## References
- Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD-AT).
- Zhang et al. (2019), *Theoretically Principled Trade-off between Robustness and Accuracy* (TRADES).
- Wang et al. (2020), *Improving Adversarial Robustness Requires Revisiting Misclassified Examples* (MART).
- Zhang et al. (2020), *Attacks Which Do Not Kill Training Make Adversarial Learning Stronger* (FAT).
- Wu et al. (2020), *Adversarial Weight Perturbation Helps Robust Generalization* (AWP).
- Loshchilov & Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts*.
- Izmailov et al. (2018), *Averaging Weights Leads to Wider Optima* (SWA / EMA).

## Files
- `train_resnet34_fatmart.ipynb` — the winning training notebook (recreates the result).
- `submission.py` — leaderboard submission script (fill in API key + model path).
- `Assignment_3_-_Robustness.pdf` — the task specification.
