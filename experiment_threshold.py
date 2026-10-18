
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data/code_final_v2.parquet"
LABEL_FILE = ROOT / "labels_all.csv"
EMBEDDING_FILE = ROOT / "embeddings_all-mpnet-base-v2.npy"

CLASSES = ["bug", "feature", "other"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def text_flags(prompt):
    head = prompt[:400].lower()
    full = prompt[:2000].lower()

    return [
        float(head.lstrip("# \n").startswith("i want")),
        float("expected outcome" in full),
        float(bool(__import__("re").search(
            r"feature request|\[feat|enhancement|proposal|"
            r"would be (nice|great)|is there (any|a) way|"
            r"can we|could we|please add|add support",
            full,
        ))),
        float(bool(__import__("re").search(
            r"traceback|exception|error|fails?|failed|crash|panic|"
            r"raises?|wrong|incorrect|broken|doesn'?t work|"
            r"regression|bug",
            full,
        ))),
        float("```" in prompt),
        float(bool(__import__("re").search(
            r"steps to reproduce|to reproduce|expected behavio|"
            r"actual behavio|what happened",
            full,
        ))),
        float(bool(__import__("re").search(
            r"[\u4e00-\u9fff]", prompt[:300]
        ))),
        len(prompt.split()) / 1000.0,
    ]


def main():
    # Load inputs.
    df = pd.read_parquet(DATA_FILE)
    labels = pd.read_csv(LABEL_FILE)
    embeddings = np.load(EMBEDDING_FILE)

    required_data = {"instance_id", "prompt"}
    required_labels = {"instance_id", "label", "source"}

    if not required_data.issubset(df.columns):
        raise ValueError(
            f"Dataset missing: {required_data - set(df.columns)}"
        )

    if not required_labels.issubset(labels.columns):
        raise ValueError(
            f"Labels missing: {required_labels - set(labels.columns)}"
        )

    df["instance_id"] = df["instance_id"].astype(str)
    labels["instance_id"] = labels["instance_id"].astype(str)
    labels["label"] = labels["label"].str.strip().str.lower()
    labels["source"] = labels["source"].str.strip().str.lower()

    if df["instance_id"].duplicated().any():
        raise ValueError("Duplicate task IDs in dataset.")

    if labels["instance_id"].duplicated().any():
        raise ValueError("Duplicate task IDs in labels.")

    if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
        raise ValueError("Embedding shape does not match dataset rows.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    row_map = {
        task_id: i for i, task_id in enumerate(df["instance_id"])
    }

    if not set(labels["instance_id"]).issubset(row_map):
        raise ValueError("Some labeled task IDs are missing from dataset.")

    rows = labels["instance_id"].map(row_map).to_numpy(dtype=int)

    if not set(labels["label"]).issubset(CLASSES):
        raise ValueError("Unexpected class labels found.")

    # Random labels are evaluation examples.
    random_mask = labels["source"].eq("random").to_numpy()
    random_rows = rows[random_mask]
    random_y = labels.loc[random_mask, "label"].to_numpy()

    # The other 100 labels are used for training only.
    extra_rows = rows[~random_mask]
    extra_y = labels.loc[~random_mask, "label"].to_numpy()

    if len(random_rows) != 250:
        print(f"Warning: expected 250 random labels, got {len(random_rows)}")

    prompts = df["prompt"].fillna("").astype(str).tolist()
    heads = [p[:400].lower() for p in prompts]
    flags = np.asarray([text_flags(p) for p in prompts], dtype=float)

    # Store out-of-fold probabilities. Each random-labeled task
    # is predicted by a model that did not train on that task.
    class_index = {label: i for i, label in enumerate(CLASSES)}
    oof = np.zeros((len(random_rows), len(CLASSES)), dtype=float)
    counts = np.zeros(len(random_rows), dtype=int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(random_rows, random_y), start=1
    ):
        train_rows = np.concatenate(
            [random_rows[train_idx], extra_rows]
        )
        y_train = np.concatenate([random_y[train_idx], extra_y])
        test_rows = random_rows[test_idx]

        # Fit the TF-IDF vocabulary on training data only.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
            dtype=np.float32,
        )
        vectorizer.fit([heads[i] for i in train_rows])

        # Fit scaling on training data only.
        scaler = StandardScaler()
        scaler.fit(flags[train_rows])

        X_train = hstack([
            vectorizer.transform([heads[i] for i in train_rows]),
            csr_matrix(embeddings[train_rows]),
            csr_matrix(0.5 * scaler.transform(flags[train_rows])),
        ], format="csr")

        X_test = hstack([
            vectorizer.transform([heads[i] for i in test_rows]),
            csr_matrix(embeddings[test_rows]),
            csr_matrix(0.5 * scaler.transform(flags[test_rows])),
        ], format="csr")

        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=5.0,
            random_state=42,
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)

        for col, label in enumerate(model.classes_):
            oof[test_idx, class_index[label]] = probabilities[:, col]

        counts[test_idx] += 1
        print(f"Completed fold {fold}/5")

    if not np.all(counts == 1):
        raise RuntimeError("Invalid out-of-fold predictions.")

    # Standard argmax baseline.
    baseline = np.asarray(CLASSES)[oof.argmax(axis=1)]

    print("\nBASELINE: STANDARD ARGMAX")
    print(classification_report(
        random_y, baseline,
        labels=CLASSES,
        zero_division=0,
    ))

    results = []

    # Threshold rule:
    # If P(other) >= threshold, predict other.
    # Otherwise choose between bug and feature.
    for threshold in THRESHOLDS:
        predictions = np.where(
            oof[:, class_index["other"]] >= threshold,
            "other",
            np.where(
                oof[:, class_index["bug"]]
                >= oof[:, class_index["feature"]],
                "bug",
                "feature",
            ),
        )

        other_precision = precision_score(
            random_y, predictions,
            labels=["other"],
            average="macro",
            zero_division=0,
        )
        other_recall = recall_score(
            random_y, predictions,
            labels=["other"],
            average="macro",
            zero_division=0,
        )
        other_f1 = f1_score(
            random_y, predictions,
            labels=["other"],
            average="macro",
            zero_division=0,
        )

        results.append({
            "other_threshold": threshold,
            "accuracy": accuracy_score(random_y, predictions),
            "macro_f1": f1_score(
                random_y, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "other_precision": other_precision,
            "other_recall": other_recall,
            "other_f1": other_f1,
            "predicted_other": int((predictions == "other").sum()),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT / "threshold_experiment.csv", index=False)

    print("\nTHRESHOLD COMPARISON")
    print(results_df.round(3).to_string(index=False))

    # Show a candidate that prioritizes macro F1.
    best = results_df.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]

    selected_threshold = float(best["other_threshold"])
    selected_predictions = np.where(
        oof[:, class_index["other"]] >= selected_threshold,
        "other",
        np.where(
            oof[:, class_index["bug"]]
            >= oof[:, class_index["feature"]],
            "bug",
            "feature",
        ),
    )

    print(f"\nExploratory best threshold: {selected_threshold:.2f}")
    print("Selected-threshold report:")
    print(classification_report(
        random_y, selected_predictions,
        labels=CLASSES,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(random_y, selected_predictions, labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    ))

    print("\nSaved: threshold_experiment.csv")
    print("No dataset or existing model files were overwritten.")


if __name__ == "__main__":
    main()
