
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack
import numpy as np

DATA_PATH = "data/code_final_v2.parquet"
LABEL_PATH = "labels_all.csv"
EMBEDDING_PATH = "embeddings_all-mpnet-base-v2.npy"

CLASSES = ["bug", "feature", "other"]
THRESHOLD = 0.15

data = pd.read_parquet(DATA_PATH).drop_duplicates("instance_id")
labels = pd.read_csv(LABEL_PATH)

# Evaluate only the randomly labeled examples.
labels = labels[labels["source"] == "random"].copy()
labels = labels[labels["label"].isin(CLASSES)].copy()

data = data.set_index("instance_id")
labels["prompt"] = labels["instance_id"].map(data["prompt"])
labels = labels.dropna(subset=["prompt"]).reset_index(drop=True)

all_embeddings = np.load(EMBEDDING_PATH)
canonical_ids = data.index.tolist()

if len(all_embeddings) != len(canonical_ids):
    raise ValueError("Embedding rows do not match dataset rows.")

embedding_map = dict(zip(canonical_ids, all_embeddings))
extra = pd.read_csv(LABEL_PATH)
extra = extra[extra["source"] != "random"]
extra = extra[extra["label"].isin(CLASSES)].copy()
extra["prompt"] = extra["instance_id"].map(data["prompt"])
extra = extra.dropna(subset=["prompt"])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_true = []
oof_pred = []
oof_ids = []
oof_prompts = []

for train_idx, test_idx in cv.split(labels, labels["label"]):
    train = labels.iloc[train_idx].copy()
    test = labels.iloc[test_idx].copy()

    train = pd.concat([train, extra], ignore_index=True)
    train = train.drop_duplicates("instance_id", keep="first")

    train_emb = np.stack([embedding_map[i] for i in train["instance_id"]])
    test_emb = np.stack([embedding_map[i] for i in test["instance_id"]])

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    x_train_text = vectorizer.fit_transform(train["prompt"])
    x_test_text = vectorizer.transform(test["prompt"])

    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    test_emb = scaler.transform(test_emb)

    x_train = hstack([x_train_text, csr_matrix(train_emb)]).tocsr()
    x_test = hstack([x_test_text, csr_matrix(test_emb)]).tocsr()

    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )
    model.fit(x_train, train["label"])

    raw = model.predict_proba(x_test)
    probs = np.zeros((len(test), len(CLASSES)))
    for j, label in enumerate(model.classes_):
        probs[:, CLASSES.index(label)] = raw[:, j]

    pred = []
    for row in probs:
        if row[CLASSES.index("other")] >= THRESHOLD:
            pred.append("other")
        else:
            pred.append(
                CLASSES[
                    np.argmax(row[[CLASSES.index("bug"),
                                   CLASSES.index("feature")]])
                    if False else 0
                ]
            )
            # Choose bug vs feature by their relative scores.
            pred[-1] = (
                "bug" if row[0] >= row[1] else "feature"
            )

    oof_true.extend(test["label"].tolist())
    oof_pred.extend(pred)
    oof_ids.extend(test["instance_id"].tolist())
    oof_prompts.extend(test["prompt"].tolist())

results = pd.DataFrame({
    "instance_id": oof_ids,
    "true_label": oof_true,
    "predicted_label": oof_pred,
    "prompt": oof_prompts,
})

results["correct"] = results["true_label"] == results["predicted_label"]

print("\nClassification report:")
print(classification_report(
    results["true_label"],
    results["predicted_label"],
    labels=CLASSES,
    zero_division=0,
))

print("\nActual 'other' examples:")
print(
    results[results["true_label"] == "other"][
        ["instance_id", "predicted_label", "correct", "prompt"]
    ].to_string(index=False, max_colwidth=180)
)

print("\nExamples predicted as 'other' but labeled bug/feature:")
print(
    results[
        (results["predicted_label"] == "other")
        & (results["true_label"] != "other")
    ][["instance_id", "true_label", "prompt"]]
    .to_string(index=False, max_colwidth=180)
)

results.to_csv("other_error_inspection.csv", index=False)
print("\nSaved: other_error_inspection.csv")
