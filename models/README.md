# models/

Placeholder directory for the trained XGBoost budget-predictor artifacts.

After running:

```bash
python -m budget_predictor.trainer \
    --cnf-dir path/to/cnf_instances \
    --results-csv best_params_results.csv \
    --model-dir models
```

this directory will contain:

| File | Description |
|---|---|
| `budget_xgb_model.pkl` | Trained `BudgetXGBModel` (XGBoost regressor, pickled) |
| `feature_scaler.json` | Mean/std used to standardize features at inference time |
| `metadata.json` | Feature names, train/test size, MAE, R², quality target used |

These files are not committed to the repository (they are generated
artifacts, and the model is only meaningful once trained on your own
profiling results). This placeholder keeps the directory tracked in
version control.
