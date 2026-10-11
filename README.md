# Daily Solar Radiation Forecasting with Neural Networks

UNSW COMP9414 (Artificial Intelligence), Term 1 2026 — Assignment 1 (due 20 March 2026), individual coursework.

**中文简介**：用神经网络预测未来 1–7 天的日太阳辐射（W/m²），包含数据预处理（傅里叶周期编码、滚动统计、坏数据剔除）、多输出 MLP 与三种自定义损失、三种迁移学习策略、MC Dropout 不确定性、对滞后特征的缩放点积注意力、消融实验与配对 t 检验。模型均为小型 MLP（没有 LSTM/Transformer）；消融结果显示注意力和滞后特征在本数据集上没有带来可测量的提升，主要信号来自季节性（傅里叶）特征。

All code, experiments and analysis in `z5565492_assignment1.ipynb` are my own work. The task specification, datasets and the hidden evaluation set are course material and are not included.

## What the notebook covers

| Part | Content |
|---|---|
| A | Preprocessing: drop bad-quality rows (1,005 of 6,575, 15.3%), leap-year-aware Fourier day-of-year encoding, 7-day rolling statistics, 7 lag features, chronological split (train 1996–2008 / val 2009–2010 / test 2011–2013) → 13 input features, 7 targets (t+1 … t+7) |
| B | Multi-output MLP (hidden 128, dropout 0.2) with three losses: weighted MSE, pseudo-Huber (δ = 50), weighted quantile (τ = 0.8) |
| C | Seven individual single-output models and three transfer strategies: sequential fine-tuning, feature extraction, multi-task learning |
| D | MC Dropout (T = 50), attention model (scaled dot-product attention over the 7 lag features), ablation study |
| E | Paired t-tests on per-day MAE, error analysis by season / weather / magnitude |

The assignment required feedforward MLP architectures.

## Results (as printed by the notebook; CPU, PyTorch 2.0.1)

Part B, macro average over the 7 horizons:

| Loss | RMSE | MAE |
|---|---|---|
| Weighted MSE | 61.25 | 47.66 |
| Pseudo-Huber | 61.45 | 47.18 |
| Quantile (τ = 0.8) | 78.36 | 56.47 |

The quantile loss is deliberately asymmetric (mean bias ≈ +46 W/m²), so its RMSE is not a like-for-like comparison.

- RMSE grows with horizon, from about 57.6 (t+1) to 65.3 (t+7) for the multi-output model.
- Part C macro RMSE: multi-output 61.25, multi-task 61.54, seven individual models ≈ 61.8, sequential fine-tuning ≈ 62.1; feature extraction alone was clearly worse (t+7 RMSE 69.78).
- On the course's hidden evaluation set (2,012 samples) the weighted-MSE model scored macro RMSE 59.24 / MAE 45.28.
- Ablation (attention model as baseline, RMSE 61.55): removing the Fourier features raises RMSE by +8.1; removing rolling statistics +0.14; removing lag features −0.32 and removing attention −0.40.
- MC Dropout: only 28–33% of observations fall within ±2σ across horizons, so the uncertainty estimates are under-dispersed.
- Paired t-tests on per-day MAE (original run): multi-output vs individual p = 8.6e-4, multi-output vs multi-task p = 1.0e-4, attention vs multi-output p = 0.39.

## Rerun check (11 October 2026)

I re-executed the whole notebook, unchanged apart from the data path, in a fresh environment (Python 3.13, torch 2.14.1, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1; CPU; 12.5 minutes). Results are close but not identical:

| Quantity | Original run | Rerun |
|---|---|---|
| Part B macro RMSE: weighted MSE / pseudo-Huber / quantile | 61.25 / 61.45 / 78.36 | 61.14 / 61.07 / 79.72 |
| Hidden set macro RMSE (weighted MSE) | 59.24 | 59.30 |
| Ablation ΔRMSE without Fourier / rolling / lags / attention | +8.14 / +0.14 / −0.32 / −0.40 | +8.98 / +0.40 / −0.07 / −0.33 |
| Paired t-test p: multi-output vs individual / vs multi-task / attention vs multi-output | 8.6e-4 / 1.0e-4 / 0.39 | 0.20 / 0.73 / 0.27 |

The large effects reproduce (Fourier features, quantile-loss bias, feature extraction being worse, under-dispersed MC Dropout). Differences of a few tenths of an RMSE between models and the t-test significance do not: each model was trained once with one seed.

## How to reproduce

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib jupyter
export SOLAR_DATA_DIR=/path/to/datasets   # daily_solar_1996.csv ... daily_solar_2013.csv
jupyter notebook z5565492_assignment1.ipynb   # Restart & Run All
```

- The datasets are course material and are not included. The notebook expects one CSV per year with columns including `date`, `solar_t` and `bad_quality`. Without `SOLAR_DATA_DIR` it looks for a `datasets/` folder next to the notebook. Path handling is the only change from the submitted notebook (it used absolute `H:\…` paths): `SOLAR_DATA_DIR` for the data and an optional `SOLAR_HIDDEN_DIR` for the hidden set (skipped if unset; its stored output is the result on the course's hidden data).
- Each model trains in roughly 4–50 s with early stopping; the whole notebook took about 12.5 minutes in the rerun. Seeds are fixed (`seed(42)`), but results still vary slightly across library versions.

## Files

- `z5565492_assignment1.ipynb` – full solution notebook (with stored outputs)
- `model_*.pt`, `scaler.pkl` – trained models and fitted scaler
- `part*_*.png`, `eval_predictions.png` – figures produced by the notebook
