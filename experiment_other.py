
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, hstack

# Files from your existing project
DATA_FILE = "data/code_final_v2.parquet"
LABEL_FILE = "labels_all.csv"
EMBEDDING_FILE = "embeddings_all-mpnet-base-v2.npy"

CLASSES = ["bug", "feature", "other"]


def main():
    df = pd.read_parquet(DATA_FILE)
    labels = pd.read_csv(LABEL_FILE)
    embeddings = np.load(EMBEDDING_FILE)

    # Align labels to dataset by stable ID, never by row order.
    df["instance_id"] = df["instance_id"].astype(str)
    labels["instance_id"] = labels["instance_id"].astype(str)
    labels["label"] = labels["label"].str.lower().str.strip()

    if df["instance_id"].duplicated().any():
        raise ValueError("Duplicate instance IDs in dataset.")
    if labels["instance_id"].duplicated().any():
        raise ValueError("Duplicate instance IDs in labels.")

    row_lookup = {
        task_id: i for i, task_id in enumerate(df["instance_id"])
    }
    if not set(labels["instance_id"]).issubset(row_lookup):
        raise ValueError("Some label IDs are missing from the dataset.")

    rows = labels["instance_id"].map(row_lookup).to_numpy()
    if embeddings.shape[0] != len(df):
        raise ValueError("Embedding rows do not match dataset rows.")

    # Use the random-label set for evaluation.
    random_mask = labels["source"].str.lower().eq("random").to_numpy()
    random_rows = rows[random_mask]
    random_y = labels.loc[random_mask, "label"].to_numpy()

    # The additional uncertain examples are training-only.
    extra_rows = rows[~random_mask]
    extra_y = labels.loc[~random_mask, "label"].to_numpy()

    prompts = df["prompt"].fillna("").astype(str).tolist()
    texts = [p[:400].lower() for p in prompts]

    # Fixed folds make model comparisons fair.
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    splits = list(cv.split(random_rows, random_y))

    # Test whether increasing the relative cost of "other"
    # helps. These are experimental weights, not guaranteed fixes.
    other_weights = [1.0, 2.0, 3.0, 4.0]

    for other_weight in other_weights:
        true_all, pred_all = [], []

        for train_idx, test_idx in splits:
            train_rows = np.concatenate(
                [random_rows[train_idx], extra_rows]
            )
            y_train = np.concatenate(
                [random_y[train_idx], extra_y]
            )
            test_rows = random_rows[test_idx]
            y_test = random_y[test_idx]

            # Fit TF-IDF only on the training fold.
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            )
            vectorizer.fit([texts[i] for i in train_rows])

            X_train = hstack([
                vectorizer.transform([texts[i] for i in train_rows]),
                csr_matrix(embeddings[train_rows]),
            ], format="csr")

            X_test = hstack([
                vectorizer.transform([texts[i] for i in test_rows]),
                csr_matrix(embeddings[test_rows]),
            ], format="csr")

            model = LogisticRegression(
                max_iter=5000,
                class_weight={
                    "bug": 1.0,
                    "feature": 1.0,
                    "other": other_weight,
                },
                C=5.0,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            true_all.extend(y_test)
            pred_all.extend(pred)

        print("\n" + "=" * 55)
        print(f"OTHER CLASS WEIGHT = {other_weight}")
        print(classification_report(
            true_all, pred_all,
            labels=CLASSES,
            zero_division=0,
        ))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(pd.DataFrame(
            confusion_matrix(true_all, pred_all, labels=CLASSES),
            index=CLASSES,
            columns=CLASSES,
        ))


if __name__ == "__main__":
    main()
