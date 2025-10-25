# scripts/train_xgb.py
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path
import datetime
import uuid

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # assume target column is 'rpc_label' or 'rpc'
    if 'rpc_label' in df.columns:
        df['rpc_label'] = df['rpc_label'].astype(int)
    elif 'rpc' in df.columns:
        df['rpc_label'] = df['rpc'].astype(int)
    else:
        raise ValueError("No rpc_label or rpc target found in data for XGBoost.")
    # fill missing numeric with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove target and cluster from num_cols? Keep cluster if exists
    if 'rpc_label' in num_cols:
        num_cols.remove('rpc_label')
    # simple preprocessing: drop id columns if any
    possible_ids = [c for c in df.columns if 'id' in c.lower()]
    df = df.drop(columns=[c for c in possible_ids if c not in ['agent_id']], errors='ignore')
    # categorical: agent_id, region, call_disposition, etc. We'll one-hot encode categorical columns with small cardinality
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # convert boolean-like columns
    df = df.copy()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('missing').astype(str)
    # One-hot encode categorical cols
    if len(cat_cols)>0:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat = ohe.fit_transform(df[cat_cols])
        cat_names = ohe.get_feature_names_out(cat_cols)
        X_cat_df = pd.DataFrame(X_cat, columns=cat_names, index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), X_cat_df], axis=1)
    return df

def train_xgb(df, out_html):
    # prepare X,y
    y = df['rpc_label']
    X = df.drop(columns=['rpc_label'])
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # small grid
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    param_grid = {
        'n_estimators':[50,100],
        'max_depth':[3,5],
        'learning_rate':[0.05, 0.1],
        'subsample':[0.8,1.0],
        'colsample_bytree':[0.8,1.0]
    }
    grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_proba = best.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    # feature importance plot
    fi = best.feature_importances_
    feat_names = X.columns
    fi_df = pd.DataFrame({'feature':feat_names, 'importance':fi}).sort_values('importance', ascending=False).head(20)
    outdir = os.path.join(os.getcwd(), 'app', 'static', 'model_plots')
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
    plt.xlabel('Importance'); plt.title('Top features (approx)')
    plot_path = os.path.join(outdir, 'xgb_feature_importance.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    # generate HTML
    html = f"""
    <html><head><title>XGBoost RPC Report</title></head><body>
    <h1>XGBoost RPC Report</h1>
    <h3>Best params</h3><pre>{grid.best_params_}</pre>
    <h3>AUC</h3><p>{auc:.4f}</p>
    <h3>Classification Report</h3><pre>{report}</pre>
    <h3>Confusion Matrix</h3><pre>{cm}</pre>
    <h3>Top features</h3>
    <img src="../static/model_plots/{os.path.basename(plot_path)}" style="max-width:700px;"><br>
    </body></html>
    """
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    # Save model artifact too
    joblib.dump(best, out_html.replace('.html', '.joblib'))
    print("XGBoost report created at", out_html)

def main():
    ap = argparse.ArgumentParser()
    # Default path: repository-relative data file
    DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic_callcenter_accounts.csv'
    ap.add_argument('--data', default=str(DEFAULT_DATA), help='path to input CSV (default: repository data)')
    ap.add_argument('--out', default=None, help='output html path (default: ml_models/xgb_<timestamp>_<id>.html)')
    args = ap.parse_args()
    df = load_data(args.data)
    df_proc = preprocess(df)
    out_html = args.out
    if not out_html:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        ml_dir = os.path.join(os.getcwd(), 'ml_models')
        os.makedirs(ml_dir, exist_ok=True)
        out_html = os.path.join(ml_dir, f'xgb_{ts}_{unique_id}.html')
    train_xgb(df_proc, out_html)

if __name__ == "__main__":
    main()
