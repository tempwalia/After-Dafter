"""

It will create:
 - ml_models/kmeans_report.html
 - ml_models/kmeans_clusters.png
 - ml_models/xgboost_report.html
 - ml_models/xgb_feature_importance.png

The Flask app will list any `.html` files found in `ml_models/` and serve them via
`/ml-models` -> link -> `/ml-models/view/<filename>`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / 'synthetic_callcenter_accounts.csv'
OUTPUT_DIR = REPO_ROOT / 'ml_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Numeric features we'll use for clustering and modeling
NUMERIC_FEATURES = [
    'avg_payment_delay', 'payment_ratio', 'missed_payments',
    'balance', 'tenure_months', 'prior_contact_rate', 'num_calls_last_30'
]

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ color: #2b7cff }}
    .section {{ margin-bottom: 30px }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px }}
    th, td {{ border: 1px solid #ddd; padding: 8px }}
    th {{ background: #f4f6f8 }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="section">
    {body}
  </div>
</body>
</html>
"""


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    return df


def generate_kmeans_report(df, n_clusters=4):
    df_clean = df.dropna(subset=NUMERIC_FEATURES + ['rpc_label']).copy()
    X = df_clean[NUMERIC_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # KMeans clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(Xs)
    df_clean['cluster'] = labels

    # PCA 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
    plt.title('KMeans clusters (2D PCA projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(False)
    plt.tight_layout()
    kmeans_img = OUTPUT_DIR / 'kmeans_clusters.png'
    plt.savefig(kmeans_img)
    plt.close()

    # Cluster summaries
    cluster_counts = df_clean.groupby('cluster').size().rename('count')
    rpc_mean = df_clean.groupby('cluster')['rpc_label'].mean().rename('rpc_rate')
    cluster_summary = pd.concat([cluster_counts, rpc_mean], axis=1).reset_index()

    # Build HTML body
    body = []
    body.append('<h2>Cluster summary</h2>')
    body.append(cluster_summary.to_html(index=False))
    body.append('<h2>Cluster scatter (PCA 2D)</h2>')
    body.append(f'<img src="{kmeans_img.name}" alt="KMeans clusters" style="max-width:900px;width:100%">')

    html = HTML_TEMPLATE.format(title='KMeans Clustering Report', body='\n'.join(body))
    out_file = OUTPUT_DIR / 'kmeans_report.html'
    out_file.write_text(html, encoding='utf-8')
    print(f'Wrote {out_file}')

    # Return df with cluster labels for downstream modeling
    return df_clean


def generate_xgboost_report(df_with_clusters):
    df_clean = df_with_clusters.dropna(subset=NUMERIC_FEATURES + ['rpc_label', 'cluster']).copy()

    # Features include numeric features + cluster
    X = df_clean[NUMERIC_FEATURES + ['cluster']]
    y = df_clean['rpc_label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Try to use XGBoost if available, else fallback to sklearn's GradientBoostingClassifier
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        # Feature importance
        try:
            fi = model.get_booster().get_score(importance_type='gain')
            # convert dict to series
            fi_series = pd.Series(fi).sort_values(ascending=False)
        except Exception:
            fi_series = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    except Exception as e:
        print('XGBoost not available, falling back to GradientBoostingClassifier:', e)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        fi_series = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    fi_series.sort_values().plot(kind='barh')
    plt.title('Feature importance')
    plt.tight_layout()
    fi_img = OUTPUT_DIR / 'xgb_feature_importance.png'
    plt.savefig(fi_img)
    plt.close()

    # Classification report
    clf_report = metrics.classification_report(y_test, y_pred, output_dict=False)

    # Build HTML body
    body = []
    body.append('<h2>Model evaluation</h2>')
    body.append(f'<p>Accuracy: {acc:.4f} &nbsp; AUC: {auc:.4f}</p>')
    body.append('<h3>Classification report</h3>')
    body.append(f'<pre>{clf_report}</pre>')
    body.append('<h2>Feature importance</h2>')
    body.append(f'<img src="{fi_img.name}" alt="Feature importance" style="max-width:900px;width:100%">')

    html = HTML_TEMPLATE.format(title='XGBoost RPC Prediction Report', body='\n'.join(body))
    out_file = OUTPUT_DIR / 'xgboost_report.html'
    out_file.write_text(html, encoding='utf-8')
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    print('Generating KMeans report...')
    df_with_clusters = generate_kmeans_report(df, n_clusters=4)
    print('Generating XGBoost report (uses cluster as feature)...')
    generate_xgboost_report(df_with_clusters)
    print('Done. Generated reports are in', OUTPUT_DIR)
