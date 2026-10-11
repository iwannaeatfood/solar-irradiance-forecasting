# Daily Solar Radiation Forecasting with Neural Networks

UNSW COMP9414 (Artificial Intelligence), Term 1 2026 — Assignment 1 (due 20 March 2026).

**中文简介**：用神经网络预测日太阳辐射（W/m²），包含数据预处理（傅里叶周期编码、滚动统计、坏数据标记处理）、多输出模型与自定义损失、迁移学习策略、MC Dropout 不确定性、注意力机制及统计分析。

## What the notebook covers

`z5565492_assignment1.ipynb` runs end to end and writes the plots and model files in this folder.

| Part | Content |
|---|---|
| A | Preprocessing: bad-quality flag handling, Fourier cyclical encoding, rolling statistics, chronological train/val/test split |
| B | Multi-output model with three loss functions: weighted MSE, pseudo-Huber, quantile |
| C | Individual per-horizon models and three transfer strategies: sequential fine-tuning, feature extraction, multi-task learning |
| D | MC Dropout (T = 50), attention model, ablation, calibration |
| E | Statistical comparison and error analysis |

## Selected results (as printed by the notebook)

Part B, macro average over horizons:

| Loss | RMSE | MAE |
|---|---|---|
| Weighted MSE | 61.25 | 47.66 |
| Pseudo-Huber | 61.45 | 47.18 |
| Quantile | 78.36 | 56.47 |

- Individual models: RMSE rises from about 58.1 W/m² at t+1 to about 64.4 W/m² at t+7.
- Part D ablation: removing the attention layer gave RMSE 61.14 vs 61.55 for the full attention model, i.e. no measurable gain from attention in this setup.
- MC Dropout calibration: only about 28–33% of observations fell within ±2σ across horizons, so the uncertainty estimates are under-dispersed.

## Files

- `z5565492_assignment1.ipynb` – full solution notebook
- `model_*.pt`, `scaler.pkl` – trained models and scaler
- `part*_*.png`, `eval_predictions.png` – figures

The course datasets and specification are not included.
