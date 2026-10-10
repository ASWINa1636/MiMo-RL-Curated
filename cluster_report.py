import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_parquet("data/code_embedded.parquet")

tf = TfidfVectorizer(max_features=8000, stop_words="english", min_df=3)
X = tf.fit_transform(df["prompt"].str.slice(0, 1500))
words = np.array(tf.get_feature_names_out())

rows = []
for c in sorted(df["cluster_id"].unique()):
    idx = np.where(df["cluster_id"] == c)[0]
    scores = np.asarray(X[idx].mean(axis=0)).ravel()
    top = ", ".join(words[scores.argsort()[::-1][:8]])
    sub = df.iloc[idx]
    rows.append({
        "cluster": c,
        "size": len(idx),
        "top_words": top,
        "top_language": sub["prog_language"].value_counts().idxmax(),
        "main_type": sub["issue_type"].value_counts().idxmax(),
        "pct_hard": round((sub["difficulty"] == "hard").mean() * 100),
    })

rep = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(rep.to_string(index=False))
rep.to_csv("cluster_report.csv", index=False)

d = df[df["near_duplicate"]]
print("\nNEAR-DUPLICATE PAIRS:", len(d))
for i, r in d.iterrows():
    print(round(r["nearest_sim"], 3))
    print(" A:", r["prompt"][:150].replace("\n", " "))
    print(" B:", df["prompt"].iloc[int(r["nearest_idx"])][:150].replace("\n", " "))