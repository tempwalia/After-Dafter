# scripts/train_kmeans.py
import argparse
import os
import datetime
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Default data path (portable). Use repository-relative path to avoid hard-coded
# absolute Windows paths and escape-sequence issues with backslashes.
DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic_callcenter_accounts.csv'
path = str(DEFAULT_DATA)
def load_data(path):
    return pd.read_csv(path)

def basic_preprocessing(df):
    # keep numeric columns useful for inventory segmentation
    # You can adapt this to your dataset's column names
    # We'll try to infer numeric columns automatically
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for clustering.")
    sub = df[num_cols].fillna(0)
    return df, sub, num_cols

def choose_k(X, max_k=8):
    inertias = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    # elbow heuristic: just pick 3 for demo; in production analyze inertia curve
    return 3, inertias

def run_kmeans(X, n_clusters):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    labels = km.fit_predict(Xs)
    return km, labels, scaler

def save_plots(df_numeric, labels, inertias, outdir):
    os.makedirs(outdir, exist_ok=True)
    # cluster size bar chart
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel("Cluster"); plt.ylabel("Count"); plt.title("Cluster sizes")
    p1 = os.path.join(outdir, "cluster_sizes.png")
    plt.tight_layout(); plt.savefig(p1); plt.close()

    # inertia elbow
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(inertias)+1), inertias, marker='o')
    plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow: Inertia vs K")
    p2 = os.path.join(outdir, "inertia.png")
    plt.tight_layout(); plt.savefig(p2); plt.close()
    return [p1, p2]

def generate_html_report(outpath, orig_df, numeric_cols, labels, km, plot_paths):
    # basic cluster summary
    orig_df['cluster'] = labels
    summary = orig_df.groupby('cluster').agg({numeric_cols[0]:'count'}).rename(columns={numeric_cols[0]:'count'}).reset_index()
    # build HTML
    html = f"""
    <html><head><title>KMeans Clustering Report</title></head><body>
    <h1>KMeans Clustering Report</h1>
    <h3>Summary</h3>
    <p>Number of clusters: {km.n_clusters}</p>
    <h3>Cluster counts</h3>
    {summary.to_html(index=False)}
    <h3>Plots</h3>
    """
    for p in plot_paths:
        html += f'<img src="../static/model_plots/{os.path.basename(p)}" style="max-width:700px;"><br>'
    html += "<h3>Sample rows with cluster</h3>"
    html += orig_df.head(50).to_html(index=False)
    html += "</body></html>"
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA), help='path to input CSV (default: repository data)')
    ap.add_argument('--out', default=None, help='output html path (default: ml_models/kmeans_<timestamp>_<id>.html)')
    args = ap.parse_args()
    df = load_data(args.data)
    orig_df, numeric_df, numeric_cols = basic_preprocessing(df)
    k, inertias = choose_k(numeric_df, max_k=8)
    km, labels, scaler = run_kmeans(numeric_df, n_clusters=k)

    # save plots into static/model_plots
    outdir = os.path.join(os.getcwd(), 'app', 'static', 'model_plots')
    plot_paths = save_plots(numeric_df, labels, inertias, outdir)

    # copy plot files into ml_models folder accessible via ../static/ in HTML
    # We'll reference images by ../static/model_plots/<name> from generated HTML in ml_models folder.

    # Determine output path: use provided --out or create a timestamped file in ml_models/
    outpath = args.out
    if not outpath:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        ml_dir = os.path.join(os.getcwd(), 'ml_models')
        os.makedirs(ml_dir, exist_ok=True)
        outpath = os.path.join(ml_dir, f'kmeans_{ts}_{unique_id}.html')

    # Generate HTML report
    generate_html_report(outpath, orig_df, numeric_cols, labels, km, plot_paths)
    print("KMeans report saved to", outpath)

if __name__ == "__main__":
    main()
