"""Make a small 'lite' version of the dataset (about 500 tasks) that runs on any laptop.
Balanced across the 20 clusters; inside each cluster it alternates between difficulty x issue-type groups.
Run from your project folder:  python make_lite.py
Writes: lite/code_lite_500.parquet, lite/code_lite_500.csv, lite/lite_summary.txt
"""
import os
import numpy as np
import pandas as pd

SIZE = 500
SEED = 42
DATA = next(p for p in ["code_final_v4.parquet", "data/code_final_v4.parquet"] if os.path.exists(p))

df = pd.read_parquet(DATA).reset_index(drop=True)
print("loaded", len(df), "tasks from", DATA)

# 1) drop one task from each near-duplicate pair, so the lite set has no obvious repeats
drop = df["near_duplicate"] & (df["instance_id"] > df["nearest_instance_id"])
pool = df[~drop].copy()
print("removed", int(drop.sum()), "near-duplicate task(s)")

# 2) equal quota per cluster
clusters = sorted(pool["cluster_name"].unique())
quota = {c: SIZE // len(clusters) for c in clusters}
for c in clusters[: SIZE - sum(quota.values())]:        # share out any remainder
    quota[c] += 1

rng = np.random.default_rng(SEED)
picked = []
for c in clusters:
    g = pool[pool["cluster_name"] == c]
    # groups = difficulty x issue type; take from each group in turn, in random order
    groups = [g[(g["difficulty"].astype(str) == d) & (g["issue_type"] == t)].sample(frac=1, random_state=int(rng.integers(1e9)))
              for d in ["easy", "medium", "hard"] for t in ["bug", "feature"]]
    groups = [x for x in groups if len(x)]
    take, i = [], 0
    while len(take) < min(quota[c], len(g)):
        x = groups[i % len(groups)]
        k = i // len(groups)
        if k < len(x):
            take.append(x.iloc[k])
        i += 1
        if i > 10_000:
            break
    picked.append(pd.DataFrame(take))

lite = pd.concat(picked).sort_values("instance_id").reset_index(drop=True)
assert lite["instance_id"].is_unique

os.makedirs("lite", exist_ok=True)
lite.to_parquet("lite/code_lite_500.parquet", index=False)
lite.drop(columns=["test_patch"]).to_csv("lite/code_lite_500.csv", index=False)   # csv without the long patch text


def share(col, full=df, part=lite):
    a = full[col].astype(str).value_counts(normalize=True)
    b = part[col].astype(str).value_counts(normalize=True)
    return pd.DataFrame({"full %": (a * 100).round(1), "lite %": (b * 100).round(1)}).fillna(0)


lines = [f"lite tasks: {len(lite)} (from {len(df)})",
         f"confident issue types in lite: {int(lite['issue_type_confident'].sum())} ({lite['issue_type_confident'].mean():.0%})",
         f"Chinese prompts in lite: {int(lite['has_chinese'].sum())}", ""]
for col in ["difficulty", "issue_type", "prog_language"]:
    lines += [f"--- {col}", share(col).to_string(), ""]
lines += ["--- tasks per cluster", lite["cluster_name"].value_counts().to_string()]
text = "\n".join(lines)
open("lite/lite_summary.txt", "w", encoding="utf-8").write(text)
print(text)