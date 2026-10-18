"""Give every task its cluster name (cluster_id -> cluster_name).
Pipeline position: after embed.py (which writes code_embedded.parquet), before finalize.py.
Run:  python name_clusters.py        Writes: code_final.parquet

The names were written by reading sample tasks from each cluster (see cluster_report.py).
They are only valid if embed.py gives the same cluster ids: KMeans(n_clusters=20, random_state=42) on the
all-MiniLM-L6-v2 embeddings. The script stops if the cluster ids look different.
"""
import pandas as pd

NAMES = {
    0: "Auth, users & permissions",
    1: "Go library features",
    2: "Cloud-native & infra (Go)",
    3: "CLI & config",
    4: "Config, paths & metadata",
    5: "Text, markup & parsing",
    6: "Python tooling",
    7: "Chinese & non-English prompts",
    8: "Git, versions & releases",
    9: "Python library features",
    10: "Infra & ops bug reports",
    11: "Frontend & UI",
    12: "Data formats & library APIs",
    13: "Retries, timeouts & connections",
    14: "JavaScript libraries & tooling",
    15: "Data science (numpy, pandas...)",
    16: "SQL & databases",
    17: "HTTP & APIs",
    18: "State, jobs & lifecycle",
    19: "Queries, filters & metrics",
}

df = pd.read_parquet("data/code_embedded.parquet")
if set(df["cluster_id"].unique()) != set(NAMES):
    raise SystemExit("cluster ids do not match the name list; nothing written")
df["cluster_name"] = df["cluster_id"].map(NAMES)
df.to_parquet("code_final.parquet", index=False)
print(df["cluster_name"].value_counts().to_string())
print("wrote code_final.parquet with", len(df), "rows")
