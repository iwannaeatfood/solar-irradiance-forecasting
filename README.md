# Daily Solar Radiation Forecasting with Neural Networks

UNSW COMP9414 (Artificial Intelligence), Term 1 2026 — Assignment 1 (due 20 March 2026), individual coursework.

**中文简介**：用神经网络预测未来 1–7 天的日太阳辐射（W/m²），包含数据预处理（傅里叶周期编码、滚动统计、坏数据剔除）、多输出 MLP 与三种自定义损失、三种迁移学习策略、MC Dropout 不确定性、对滞后特征的缩放点积注意力、消融实验与配对 t 检验。模型都是小型 MLP（没有 LSTM/Transformer）；消融结果显示注意力和滞后特征对本数据集没有带来可测量的提升，主要信号来自季节性（傅里叶）特征，这一点在下文如实说明。

## My contribution

Individual assignment: all code, experiments and analysis in `z5565492_assignment1.ipynb` are mine. The task specification, datasets and the hidden evaluation set are course material and are not included.

## What the notebook covers

| Part | Content |
|---|---|
| A | Preprocessing: remove rows flagged bad-quality (1,005 of 6,575 rows, 15.3%), leap-year-aware Fourier day-of-year encoding, 7-day rolling statistics, 7 lag features, chronological split (train 1996–2008 / val 2009–2010 / test 2011–2013) → 13 input features, 7 targets (t+1 … t+7) |
| B | Multi-output MLP (hidden 128, dropout 0.2) trained with three losses: weighted MSE, pseudo-Huber (δ = 50), weighted quantile (τ = 0.8) |
| C | Seven individual single-output models and three transfer strategies: sequential fine-tuning, feature extraction, multi-task learning |
| D | MC Dropout (T = 50 passes), attention model (scaled dot-product attention over the 7 lag features), ablation study |
| E | Paired t-tests on per-day MAE, error analysis by season / weather / magnitude |

## Selected results (as printed by the notebook; CPU, PyTorch 2.0.1)

Part B, macro average over the 7 horizons:

| Loss | RMSE | MAE |
|---|---|---|
| Weighted MSE | 61.25 | 47.66 |
| Pseudo-Huber | 61.45 | 47.18 |
| Quantile (τ = 0.8) | 78.36 | 56.47 |

The quantile loss is deliberately asymmetric, so its mean bias is large (MBE ≈ +46 W/m²) and RMSE is worse; it is not a like-for-like competitor on RMSE.

- RMSE grows with horizon: about 57.6 (t+1) → 65.3 (t+7) for the multi-output model.
- Part C (macro RMSE over horizons): multi-output 61.25, multi-task 61.54, seven individual models ≈ 61.8, sequential fine-tuning ≈ 62.1 (the individual t+7 model is the one case that beats multi-output: 64.3 vs 65.3). Feature extraction alone was clearly worse (t+7 RMSE 69.78).
- On the course's hidden evaluation set (2,012 samples) the weighted-MSE model scored macro RMSE 59.24 / MAE 45.28, no worse than the 61.25 / 47.66 on my own test split, so there is no sign of overfitting to my split.
- Ablation (attention model as baseline, RMSE 61.55): **removing the Fourier features raises RMSE by +8.1**; removing rolling statistics +0.14; removing the lag features −0.32 and removing attention −0.40 (both slightly *better*). So seasonality carries the signal and attention over lag features gave no measurable gain here.
- Paired t-tests on per-day MAE (original run): multi-output vs individual models p = 8.6e-4; multi-output vs multi-task p = 1.0e-4; attention vs multi-output p = 0.39. **These did not reproduce — see the rerun check below.**
- MC Dropout calibration: only 28–33% of observations fall within ±2σ across horizons (a calibrated Gaussian would give ≈95%), so the uncertainty estimates are under-dispersed.

## Rerun check (11 October 2026) — what is robust and what is not

I re-executed the whole notebook, unchanged apart from the data path, in a fresh virtual environment (Python 3.13, torch 2.14.1, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1; CPU; 12.5 minutes). The original outputs above came from PyTorch 2.0.1. Seeds are fixed, yet the numbers are not identical:

| Quantity | Original run | Rerun |
|---|---|---|
| Part B macro RMSE: weighted MSE / pseudo-Huber / quantile | 61.25 / 61.45 / 78.36 | 61.14 / 61.07 / 79.72 |
| Hidden set macro RMSE (weighted MSE) | 59.24 | 59.30 |
| Ablation: ΔRMSE without Fourier / rolling / lags / attention | +8.14 / +0.14 / −0.32 / −0.40 | +8.98 / +0.40 / −0.07 / −0.33 |
| MC Dropout ±2σ coverage (t+1 … t+6) | 32% … 30% | 32% … 27% |
| Paired t-test p: multi-output vs individual / multi-output vs multi-task / attention vs multi-output | 8.6e-4 / 1.0e-4 / 0.39 | 0.20 / 0.73 / 0.27 |

What I conclude from it:

- **Robust:** the large effects — removing the Fourier day-of-year features costs about +8 to +9 RMSE; the quantile loss (τ = 0.8) is clearly worse on RMSE because it is biased by design; feature extraction alone is worse than fine-tuning; MC Dropout coverage is far below 95%.
- **Not robust:** differences of a few tenths of an RMSE between models (multi-output vs individual vs multi-task vs attention), the sign of the "no lags" and "no attention" ablation deltas, and **the statistical significance of the t-tests**. Each model is trained once with one seed, so these differences are within retraining noise. The statement "multi-output is significantly better than individual models" holds only for the original run and should not be relied on.

## How to reproduce

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib jupyter
export SOLAR_DATA_DIR=/path/to/datasets   # folder containing daily_solar_1996.csv ... daily_solar_2013.csv
jupyter notebook z5565492_assignment1.ipynb   # Restart & Run All
```

- The datasets are course material and are not in this repository. The notebook expects one CSV per year with columns including `date`, `solar_t` and `bad_quality`. If `SOLAR_DATA_DIR` is unset it looks for a `datasets/` folder next to the notebook. Path handling is the only change from the submitted notebook (it originally used absolute `H:\…` paths): `SOLAR_DATA_DIR` for the data and, optionally, `SOLAR_HIDDEN_DIR` for the course's hidden evaluation set (skipped if unset).
- Run time on CPU is modest: each model trains in roughly 4–50 s with early stopping and the whole notebook took about 12.5 minutes in my rerun. Seeds are fixed (`seed(42)`), but results still vary slightly across library versions (see the rerun check above).
- The cells at the end of the notebook evaluate the saved `model_*.pt` files and `scaler.pkl` on an evaluation folder (as the course's marking harness did); without `SOLAR_HIDDEN_DIR` the hidden-set cell is skipped, and its stored output is the result I got on the course's hidden data.

## Design notes and likely interview questions

**Why MLPs and not LSTM/Transformer?** The assignment required a feedforward MLP architecture. The problem is also tabular after feature engineering (13 features → 7 daily targets) with only ~4.7k training rows, and the seven lags are fed in explicitly; seasonal structure comes from the Fourier features rather than recurrence. I did not try recurrent models, so I cannot say how they would compare.

**What did the attention model do, and did it help?** It projects each of the 7 lag values to a 16-dimensional embedding, applies scaled dot-product self-attention over the lags and concatenates the result with the non-lag features before an MLP head. It was used to answer "does attention over lags help?" and the ablation says no (RMSE 61.55 with attention vs 61.14 without; t-test p = 0.39).

**Why Fourier features instead of day-of-year as a number?** Day 365 and day 1 are adjacent; sin/cos of the day fraction (with leap-year denominators) makes the encoding continuous. Removing them is the largest degradation in the ablation (+8.1 RMSE).

**Why chronological splitting?** To avoid leaking future information through the lags and rolling statistics; random splitting would inflate accuracy.

**What does the t-test show and what are its limits?** It compares per-day mean MAE of two models on the same test days (paired). Days are autocorrelated, so the independence assumption is only approximate, and each model comes from a single training run — when I re-ran the notebook the significant results (p ≈ 1e-3, 1e-4) became non-significant (p ≈ 0.2, 0.7). The honest summary is that the models are statistically indistinguishable here; to claim otherwise I would need several seeds per model.

**What would I improve?** Train several seeds per model and report mean ± spread before running significance tests, calibrate the MC Dropout uncertainty (temperature scaling or an ensemble), add exogenous variables (cloud cover, pressure) instead of only history, and compare against a persistence/climatology baseline — the notebook has no such baseline.

## Files

- `z5565492_assignment1.ipynb` – full solution notebook (with stored outputs)
- `model_*.pt`, `scaler.pkl` – trained models and fitted scaler
- `part*_*.png`, `eval_predictions.png` – figures produced by the notebook
