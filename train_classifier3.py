"""Final classifier: bug vs feature, with an honest confidence score.
Run from your project folder. Needs:
  data/code_final_v2.parquet, embeddings_all-mpnet-base-v2.npy,
  labels_all.csv (ORIGINAL labels), labels_all_corrected_plus_candidates.csv
Writes: code_final_v4.parquet
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = "data/code_final_v2.parquet" if os.path.exists("data/code_final_v2.parquet") else "code_final_v2.parquet"
EMB_FILE = os.environ.get("EMB_FILE", "embeddings_all-mpnet-base-v2.npy")
CONFIDENT_T = 0.70                  # "confident" = probability >= 0.70 (about 94% accurate in cross-validation)

df = pd.read_parquet(DATA).drop_duplicates("instance_id").reset_index(drop=True)
emb = np.load(EMB_FILE)
assert len(emb) == len(df), "embeddings and data do not match"
pos = {iid: i for i, iid in enumerate(df["instance_id"])}
HEAD = [p[:400].lower() for p in df["prompt"]]


def flags(p):
    h, full = p[:400].lower(), p[:2000].lower()
    return [
        h.lstrip("# \n").startswith("i want"),
        "expected outcome" in full,
        bool(re.search(r"feature request|\[feat|enhancement|proposal|would be (nice|great)|is there (any|a) way|can we|could we|please add|add support", full)),
        bool(re.search(r"traceback|exception|error|fails?|failed|crash|panic|raises?|wrong|incorrect|broken|doesn'?t work|regression|bug", full)),
        "```" in p,
        bool(re.search(r"steps to reproduce|to reproduce|expected behavio|actual behavio|what happened", full)),
        bool(re.search(r"[\u4e00-\u9fff]", p[:300])),
        len(p.split()) / 1000,
    ]


FL = np.array([flags(p) for p in df["prompt"]], float)


def build(train_rows, other_rows):
    sc = StandardScaler().fit(FL[train_rows])
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([HEAD[i] for i in train_rows])
    f = lambda r: hstack([v.transform([HEAD[i] for i in r]), csr_matrix(emb[r]),
                          csr_matrix(0.5 * sc.transform(FL[r]))]).tocsr()
    return f(train_rows), f(other_rows)


def model():
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=5.0)


orig = pd.read_csv("labels_all.csv")
plus = pd.read_csv("labels_all_corrected_plus_candidates.csv")

# TEST rows = random tasks whose ORIGINAL label is bug or feature (truth = original label,
# so label corrections can never inflate the score)
test = orig[(orig["source"] == "random") & orig["label"].isin(["bug", "feature"])]
R = test["instance_id"].map(pos).values
yR = np.array(test["label"].tolist())
# EXTRA training rows = every other bug/feature label we have (corrected + candidates + uncertain)
extra = plus[plus["label"].isin(["bug", "feature"]) & ~plus["instance_id"].isin(test["instance_id"])]
X_ = extra["instance_id"].map(pos).values
yX = np.array(extra["label"].tolist())
print(f"test tasks (clear bug/feature): {len(R)} | extra training labels: {len(X_)}")

# ---- honest out-of-fold predictions: 3 repeats x 5 folds, pooled ----
conf, hit = [], []
for rep in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=200 + rep).split(R, yR):
        a, b = build(np.concatenate([R[tr], X_]), R[te])
        m = model().fit(a, np.concatenate([yR[tr], yX]))
        pr = m.predict_proba(b)
        conf += list(pr.max(1)); hit += list(m.classes_[pr.argmax(1)] == yR[te])
conf, hit = np.array(conf), np.array(hit)
print(f"\nBUG-vs-FEATURE ACCURACY (out-of-fold): {hit.mean():.3f}")
print("confidence >= t : share of tasks kept, accuracy on those")
for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    k = conf >= t
    print(f"  {t:.2f}: kept {k.mean():5.1%}, accuracy {hit[k].mean():.3f}")
best_t = CONFIDENT_T
print(f"-> 'confident' threshold used: {best_t}")

# ---- final model on all bug/feature labels -> every task ----
all_rows = np.concatenate([R, X_]); all_y = np.concatenate([yR, yX])
a, allX = build(all_rows, np.arange(len(df)))
final = model().fit(a, all_y)
pr = final.predict_proba(allX)
df["issue_type_rules"] = df["issue_type"] if "issue_type" in df.columns else None
df["issue_type"] = final.classes_[pr.argmax(1)]
df["issue_type_confidence"] = pr.max(1).round(3)
df["issue_type_confident"] = df["issue_type_confidence"] >= best_t
df.to_parquet("code_final_v4.parquet", index=False)
print("\nissue_type:", df["issue_type"].value_counts().to_dict())
print("confident:", int(df["issue_type_confident"].sum()), "of", len(df))