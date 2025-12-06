# ReneWind — Wind Turbine Failure Prediction

Build and tune classification models to identify turbine generator failures early so repairs can be scheduled before breakdowns, reducing maintenance costs.

## Overview
- Goal: Predict failure (`Target=1`) vs no failure (`Target=0`) from 40 anonymized sensor-derived features (`V1`–`V40`).
- Data: 20,000 labeled training rows and 5,000 labeled test rows (class-imbalanced).
- Workflow: Explore data, train and tune classifiers, evaluate on the held-out test set, and prioritize metrics aligned with business costs (high recall for failures).

## Repository Contents
- `INN_ReneWind_Main_Project_FullCode_Notebook.ipynb` — Main end-to-end notebook.
- `INN_ReneWind_Main_Project_FullCode_Notebook.html` — Static HTML export of the notebook.
- `Train.csv` — Training data (40 features + `Target`).
- `Test.csv` — Test data (40 features + `Target`, for final evaluation only).

## Data Description
- Features: `V1` … `V40` (numeric, ciphered/anonymized).
- Target: Binary label — `1` = failure, `0` = no failure.
- Shapes:
  - Train: 20,000 rows × 41 columns (40 features + `Target`).
  - Test: 5,000 rows × 41 columns (40 features + `Target`).
- Class balance (Train): Highly imbalanced (about 1,110 positives vs 18,890 negatives).

Implications: Use stratified splits, consider class weighting or resampling, and focus on recall/precision and F1 for the positive class. Threshold tuning can improve business outcomes given asymmetric costs (FN >> FP).

## Quickstart (Notebook)
1. Open `INN_ReneWind_Main_Project_FullCode_Notebook.ipynb` in Jupyter or Colab.
2. Run the first cell that installs exact package versions.
3. Restart the kernel/runtime as noted in the notebook, then run all cells sequentially.
4. The notebook loads `Train.csv`/`Test.csv`, trains models, and reports metrics (e.g., confusion matrix, precision/recall/F1) for evaluation.

## Environment and Dependencies
The notebook pins versions to ensure compatibility. If you prefer installing manually in a local environment (Python 3.9–3.11 recommended):

```
pip install --no-deps \
  tensorflow==2.18.0 \
  scikit-learn==1.3.2 \
  matplotlib==3.8.3 \
  seaborn==0.13.2 \
  numpy==1.26.4 \
  pandas==2.2.2
```

Notes:
- If you use the install cell inside the notebook, follow its instruction to restart the kernel before proceeding.
- Depending on your platform, you may prefer a virtual environment (e.g., `python -m venv .venv && . .venv/bin/activate`).

## Modeling Notes
- Start with stratified train/validation splits.
- Try class-weighted classifiers and/or resampling for imbalance.
- Evaluate using precision/recall, F1 (positive class), and confusion matrix.
- Consider threshold tuning to balance FN vs FP costs.

## Reproducibility
- Fix random seeds where applicable (NumPy, scikit-learn, TensorFlow) when comparing models.
- Document any data preprocessing (imputation, scaling) applied during training and ensure it is applied consistently at evaluation.

## Next Steps
- If desired, convert the notebook into a Python script/CLI that trains, saves the best model, and evaluates on the test set.
- Add `requirements.txt` and simple make/CLI targets for repeatable runs.

## License and Acknowledgments
Data and problem context adapted from a ciphered wind-energy maintenance dataset for educational purposes.
