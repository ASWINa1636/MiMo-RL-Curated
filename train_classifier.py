import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_parquet("data/code_final_v2.parquet")
emb = np.load("embeddings.npy")
labels = pd.read_csv("labels_all.csv")

pos = {iid: i for i, iid in enumerate(df["instance_id"])}
idx = labels["instance_id"].map(pos).values
X, y = emb[idx], np.array(labels["label"].astype(str).tolist())
print("labeled rows:", len(y)); print(pd.Series(y).value_counts().to_string())

clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pred = cross_val_predict(clf, X, y, cv=cv)
print("\nCLASSIFIER accuracy (5-fold):", round(accuracy_score(y, pred), 3))
print(classification_report(y, pred, zero_division=0))

rule = np.array(df["issue_type"].iloc[idx].replace({"crash": "bug"}).astype(str).tolist())
print("KEYWORD RULES accuracy:      ", round(accuracy_score(y, rule), 3))

clf.fit(X, y)
proba = clf.predict_proba(emb)
df["issue_type_rules"] = df["issue_type"]
df["issue_type"] = np.array(clf.classes_)[proba.argmax(axis=1)]
df["issue_type_confidence"] = proba.max(axis=1).round(3)
df.to_parquet("data/code_final_v3.parquet", index=False)
print("\nnew issue_type counts:"); print(df["issue_type"].value_counts().to_string())
print("low-confidence (<0.5):", (df["issue_type_confidence"] < 0.5).sum())