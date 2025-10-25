Model reports generator

This folder contains the script `generate_reports.py` which produces HTML reports for KMeans clustering and XGBoost RPC prediction based on `synthetic_callcenter_accounts.csv`.

How to generate reports

From the repository root run:

```powershell
python ml_models/generate_reports.py
```

Outputs (saved into `ml_models/`):
- `kmeans_report.html` and `kmeans_clusters.png`
- `xgboost_report.html` and `xgb_feature_importance.png`

Notes
- The XGBoost model will use `cluster` as an input feature (clusters come from KMeans).
- If `xgboost` is not available the generator falls back to scikit-learn's `GradientBoostingClassifier`.
- To install ML extras quickly (recommended in a virtualenv):

```powershell
pip install -r requirements-ml.txt
```

Serving reports in the app

The Flask app lists `.html` files found in `ml_models/` on the ML Models page and serves them through `/ml-models/view/<filename>` where users can inspect the generated HTML reports.
