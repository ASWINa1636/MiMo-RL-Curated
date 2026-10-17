"""Give the 20 clusters better names (based on reading 7 sample tasks from each).
Run from your project folder:  python rename_clusters.py
Renames cluster_name in code_final_v4.parquet and in data/code_final_v2.parquet (the source file),
so re-running train_classifier3.py will not bring the old names back.
"""
import os
import pandas as pd

NEW = {
    "Auth & tokens":            "Auth, users & permissions",
    "Bytes & JSON (Go)":        "Data formats & library APIs",
    "CLI & config":             "CLI & config",
    "Chinese-language prompts": "Chinese & non-English prompts",
    "Frontend":                 "Frontend & UI",
    "Git & releases":           "Git, versions & releases",
    "Go libraries":             "Go library features",
    "HTTP & APIs":              "HTTP & APIs",
    "Home Assistant":           "Infra & ops bug reports",
    "JS / React / Jest":        "JavaScript libraries & tooling",
    "Jobs & workflows":         "State, jobs & lifecycle",
    "Kubernetes":               "Cloud-native & infra (Go)",
    "Networking":               "Retries, timeouts & connections",
    "Paths & metadata":         "Config, paths & metadata",
    "Python API features":      "Python library features",
    "Python tooling":           "Python tooling",
    "Queries & metrics":        "Queries, filters & metrics",
    "SQL tools":                "SQL & databases",
    "Text formatting":          "Text, markup & parsing",
    "numpy & pandas":           "Data science (numpy, pandas...)",
}

done = 0
for path in ["code_final_v4.parquet", "data/code_final_v2.parquet", "code_final_v2.parquet"]:
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    old = set(df["cluster_name"].unique())
    if old <= set(NEW.values()):
        print(f"{path}: already renamed, skipped")
        continue
    unknown = old - set(NEW) - set(NEW.values())
    if unknown:
        raise SystemExit(f"{path}: unexpected cluster names {sorted(unknown)} - nothing changed")
    df["cluster_name"] = df["cluster_name"].replace(NEW)
    df.to_parquet(path, index=False)
    print(f"{path}: renamed, {df['cluster_name'].nunique()} clusters")
    done += 1
print("done" if done else "nothing to do")