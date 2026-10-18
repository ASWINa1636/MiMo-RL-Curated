
"""
MiMo-RL Curated — Improved issue-type classifier

Compares:
  1. MiniLM embeddings (384 dimensions)
  2. MPNet embeddings (768 dimensions)

Features:
  - TF-IDF word n-grams
  - Semantic embeddings
  - Simple text features
  - Human audit corrections when available
  - Repeated stratified CV on the random-label evaluation set
  - Per-class metrics and confidence/coverage analysis

Run from any working directory:
    python data/train_classifier_improved.py

Dependencies:
    pandas, numpy, scipy, scikit-learn, pyarrow
"""

from pathlib import Path
import re
import warnings

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
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data/code_final_v2.parquet"
LABEL_FILE = BASE_DIR / "labels_all.csv"
AUDIT_FILE = BASE_DIR / "audit_answers.csv"

OUTPUT_FILE = BASE_DIR / "code_final_v4.parquet"
METRICS_FILE = BASE_DIR / "classifier_comparison.csv"

EMBEDDING_FILES = {
    "MiniLM": BASE_DIR / "embeddings.npy",
    "MPNet": BASE_DIR / "embeddings_all-mpnet-base-v2.npy",
}

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 6

# Macro F1 is the primary selection metric because "other"
# is underrepresented. Accuracy is also reported.
CLASS_WEIGHT = "balanced"
C_VALUE = 5.0


# ---------------------------------------------------------
# 2. Load and validate data
# ---------------------------------------------------------

def load_data():
    for path in (DATA_FILE, LABEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_parquet(DATA_FILE)
    labels = pd.read_csv(LABEL_FILE)

    required_data_columns = {"instance_id", "prompt", "issue_type"}
    required_label_columns = {"instance_id", "label", "source"}

    if not required_data_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset missing columns: "
            f"{required_data_columns - set(df.columns)}"
        )

    if not required_label_columns.issubset(labels.columns):
        raise ValueError(
            f"Label file missing columns: "
            f"{required_label_columns - set(labels.columns)}"
        )

    if df["instance_id"].duplicated().any():
        raise ValueError("Duplicate instance_id values in dataset.")

    if labels["instance_id"].duplicated().any():
        raise ValueError(
            "Duplicate instance_id values in labels_all.csv. "
            "Resolve duplicates before training."
        )

    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["instance_id"] = df["instance_id"].astype(str)
    labels["instance_id"] = labels["instance_id"].astype(str)
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    labels["source"] = labels["source"].astype(str).str.strip().str.lower()

    allowed_labels = {"bug", "feature", "other"}
    unexpected = set(labels["label"]) - allowed_labels

    if unexpected:
        raise ValueError(f"Unexpected labels found: {unexpected}")

    positions = pd.Series(
        np.arange(len(df)),
        index=df["instance_id"],
    )

    missing_ids = sorted(set(labels["instance_id"]) - set(positions.index))

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} labeled IDs are absent from the dataset. "
            f"Examples: {missing_ids[:5]}"
        )

    # Map every label to the correct task ID, never by CSV row order.
    rows = labels["instance_id"].map(positions).to_numpy(dtype=int)

    return df, labels, rows


# ---------------------------------------------------------
# 3. Apply reviewed audit corrections
# ---------------------------------------------------------

def apply_audit(labels):
    labels = labels.copy()
    labels["audit_applied"] = False

    if not AUDIT_FILE.exists():
        print("\nNo audit_answers.csv found; using original labels.")
        return labels

    audit = pd.read_csv(AUDIT_FILE)

    required = {"instance_id", "human_label"}
    if not required.issubset(audit.columns):
        print("\nAudit file has no recognized human_label column.")
        return labels

    audit["instance_id"] = audit["instance_id"].astype(str)
    audit["human_label"] = (
        audit["human_label"].astype(str).str.strip().str.lower()
    )

    audit = audit.dropna(subset=["instance_id", "human_label"])

    allowed = {"bug", "feature", "other"}
    audit = audit[audit["human_label"].isin(allowed)]

    # Conflicting decisions for the same task must be resolved manually.
    conflicts = audit.groupby("instance_id")["human_label"].nunique()
    conflicts = conflicts[conflicts > 1]

    if not conflicts.empty:
        raise ValueError(
            "Conflicting human audit labels found for task IDs: "
            f"{conflicts.index.tolist()[:10]}"
        )

    audit = audit.drop_duplicates("instance_id", keep="last")
    corrections = audit.set_index("instance_id")["human_label"]

    matching = labels["instance_id"].isin(corrections.index)
    previous = labels.loc[matching, "label"].copy()

    labels.loc[matching, "label"] = (
        labels.loc[matching, "instance_id"].map(corrections)
    )
    labels.loc[matching, "audit_applied"] = True

    changed = int(
        (previous.to_numpy() != labels.loc[matching, "label"].to_numpy()).sum()
    )

    print(f"\nAudit entries: {len(audit)}")
    print(f"Audit entries matching training labels: {int(matching.sum())}")
    print(f"Labels actually changed: {changed}")
    print(
        "Audit entries not present in labels_all.csv:",
        int((~audit["instance_id"].isin(labels["instance_id"])).sum()),
    )

    return labels


# ---------------------------------------------------------
# 4. Extract text features
# ---------------------------------------------------------

def extract_flags(prompt):
    head = prompt[:400].lower()
    full = prompt[:2000].lower()

    return [
        float(head.lstrip("# \n").startswith("i want")),
        float("expected outcome" in full),
        float(bool(re.search(
            r"feature request|\[feat|enhancement|proposal|"
            r"would be (nice|great)|is there (any|a) way|"
            r"can we|could we|please add|add support",
            full,
        ))),
        float(bool(re.search(
            r"traceback|exception|error|fails?|failed|crash|panic|"
            r"raises?|wrong|incorrect|broken|doesn'?t work|"
            r"regression|bug",
            full,
        ))),
        float("```" in prompt),
        float(bool(re.search(
            r"steps to reproduce|to reproduce|expected behavio|"
            r"actual behavio|what happened",
            full,
        ))),
        float(bool(re.search(r"[\u4e00-\u9fff]", prompt[:300]))),
        len(prompt.split()) / 1000.0,
    ]


def build_features(df, embeddings, train_rows, predict_rows):
    prompts = df["prompt"].tolist()
    heads = [p[:400].lower() for p in prompts]

    flags = np.asarray([extract_flags(p) for p in prompts], dtype=float)

    # Fit all transformations only on the training rows.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
        dtype=np.float32,
    )

    vectorizer.fit([heads[i] for i in train_rows])

    scaler = StandardScaler()
    scaler.fit(flags[train_rows])

    def transform(rows):
        text_matrix = vectorizer.transform([heads[i] for i in rows])
        flag_matrix = csr_matrix(
            0.5 * scaler.transform(flags[rows])
        )
        embedding_matrix = csr_matrix(embeddings[rows])

        return hstack(
            [text_matrix, embedding_matrix, flag_matrix],
            format="csr",
        )

    return transform(train_rows), transform(predict_rows)


# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------

def new_model():
    return LogisticRegression(
        max_iter=5000,
        class_weight=CLASS_WEIGHT,
        C=C_VALUE,
        random_state=RANDOM_STATE,
    )


def evaluate_embedding(name, embeddings, df, labels, rows):
    random_mask = labels["source"].eq("random").to_numpy()

    random_rows = rows[random_mask]
    random_labels = labels.loc[random_mask, "label"].to_numpy()

    extra_rows = rows[~random_mask]
    extra_labels = labels.loc[~random_mask, "label"].to_numpy()

    if len(random_rows) < N_SPLITS * 3:
        raise ValueError("Too few random-labeled examples for CV.")

    print("\n" + "=" * 65)
    print(f"EVALUATING: {name}")
    print("=" * 65)

    print(f"Random evaluation labels: {len(random_rows)}")
    print(f"Additional training labels: {len(extra_rows)}")
    print("Label distribution in random evaluation set:")
    print(pd.Series(random_labels).value_counts().to_string())

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    accuracy_scores = []
    macro_f1_scores = []

    for train_idx, test_idx in splitter.split(random_rows, random_labels):
        # Additional uncertain examples are training-only.
        train_rows = np.concatenate([random_rows[train_idx], extra_rows])
        train_labels = np.concatenate(
            [random_labels[train_idx], extra_labels]
        )
        test_rows = random_rows[test_idx]
        test_labels = random_labels[test_idx]

        X_train, X_test = build_features(
            df, embeddings, train_rows, test_rows
        )

        model = new_model()
        model.fit(X_train, train_labels)
        predictions = model.predict(X_test)

        accuracy_scores.append(
            accuracy_score(test_labels, predictions)
        )
        macro_f1_scores.append(
            f1_score(
                test_labels,
                predictions,
                labels=["bug", "feature", "other"],
                average="macro",
                zero_division=0,
            )
        )

    # One out-of-fold prediction per random-labeled example.
    # Used for per-class metrics and confidence/coverage analysis.
    oof = np.zeros((len(random_rows), 3), dtype=float)
    class_names = ["bug", "feature", "other"]
    class_to_col = {label: i for i, label in enumerate(class_names)}

    oof_counts = np.zeros(len(random_rows), dtype=int)

    single_splitter = list(
        # Five folds; no repeated predictions in this evaluation.
        __import__("sklearn.model_selection", fromlist=["StratifiedKFold"])
        .StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        .split(random_rows, random_labels)
    )

    for train_idx, test_idx in single_splitter:
        train_rows = np.concatenate([random_rows[train_idx], extra_rows])
        train_labels = np.concatenate(
            [random_labels[train_idx], extra_labels]
        )

        X_train, X_test = build_features(
            df, embeddings, train_rows, random_rows[test_idx]
        )

        model = new_model()
        model.fit(X_train, train_labels)
        probabilities = model.predict_proba(X_test)

        for model_col, label in enumerate(model.classes_):
            oof[test_idx, class_to_col[label]] = probabilities[:, model_col]

        oof_counts[test_idx] += 1

    if not np.all(oof_counts == 1):
        raise RuntimeError("Invalid out-of-fold prediction counts.")

    predictions = np.asarray(class_names)[oof.argmax(axis=1)]
    confidence = oof.max(axis=1)

    print(
        f"\nRepeated CV accuracy: "
        f"{np.mean(accuracy_scores):.3f} "
        f"+/- {np.std(accuracy_scores):.3f}"
    )
    print(
        f"Repeated CV macro F1: "
        f"{np.mean(macro_f1_scores):.3f} "
        f"+/- {np.std(macro_f1_scores):.3f}"
    )

    print("\nOut-of-fold classification report:")
    print(classification_report(
        random_labels,
        predictions,
        labels=class_names,
        zero_division=0,
    ))

    print("Out-of-fold confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(
            random_labels,
            predictions,
            labels=class_names,
        ),
        index=class_names,
        columns=class_names,
    ))

    print("\nRaw confidence threshold analysis:")
    for threshold in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = confidence >= threshold

        if keep.sum() == 0:
            print(f"{threshold:.1f}: no predictions retained")
            continue

        acc = accuracy_score(random_labels[keep], predictions[keep])

        print(
            f"{threshold:.1f}: "
            f"coverage={keep.mean():.1%}, "
            f"accuracy={acc:.3f}, "
            f"n={keep.sum()}"
        )

    return {
        "embedding": name,
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
        "cv_accuracy_std": float(np.std(accuracy_scores)),
        "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(macro_f1_scores)),
    }


# ---------------------------------------------------------
# 6. Train final model and generate predictions
# ---------------------------------------------------------

def train_final(name, embeddings, df, labels, rows):
    print(f"\nTraining final model with {name} embeddings...")

    X_train, X_all = build_features(
        df,
        embeddings,
        rows,
        np.arange(len(df)),
    )

    model = new_model()
    model.fit(X_train, labels["label"].to_numpy())

    probabilities = model.predict_proba(X_all)
    predicted_labels = model.classes_[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)

    output = df.copy()

    # Preserve the previous rule-based classification for comparison.
    output["issue_type_rules"] = output["issue_type"]
    output["issue_type"] = predicted_labels
    output["issue_type_confidence"] = confidence.round(4)

    # These are raw model scores, not calibrated probabilities.
    output["issue_type_confident"] = confidence >= 0.80
    output["issue_type_tier"] = pd.cut(
        confidence,
        bins=[-np.inf, 0.60, 0.80, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)

    output.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved predictions to: {OUTPUT_FILE}")
    print("\nPredicted issue-type counts:")
    print(output["issue_type"].value_counts().to_string())
    print(
        "\nHigh-score predictions (raw score >= 0.80): "
        f"{int(output['issue_type_confident'].sum())} / {len(output)}"
    )


# ---------------------------------------------------------
# 7. Main
# ---------------------------------------------------------

def main():
    df, labels, rows = load_data()
    labels = apply_audit(labels)

    random_mask = labels["source"].eq("random").to_numpy()
    print(f"\nDataset tasks: {len(df)}")
    print(f"Labeled examples: {len(labels)}")
    print(f"Random evaluation labels: {int(random_mask.sum())}")
    print(f"Additional training labels: {int((~random_mask).sum())}")

    results = []
    available_embeddings = {}

    for name, path in EMBEDDING_FILES.items():
        if not path.exists():
            print(f"\nSkipping {name}; file not found: {path}")
            continue

        embeddings = np.load(path, mmap_mode="r")

        if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
            raise ValueError(
                f"{name} embedding shape {embeddings.shape} does not match "
                f"{len(df)} dataset rows. Verify row order and IDs."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(f"{name} embeddings contain NaN or infinity.")

        print(f"\n{name} embedding shape: {embeddings.shape}")
        available_embeddings[name] = embeddings

        result = evaluate_embedding(
            name, embeddings, df, labels, rows
        )
        results.append(result)

    if not results:
        raise FileNotFoundError("No embedding files were available.")

    metrics = pd.DataFrame(results).sort_values(
        "cv_macro_f1_mean",
        ascending=False,
    )
    metrics.to_csv(METRICS_FILE, index=False)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(metrics.to_string(index=False))
    print(f"\nSaved comparison: {METRICS_FILE}")

    best_name = metrics.iloc[0]["embedding"]
    print(f"\nSelected by mean CV macro F1: {best_name}")

    train_final(
        best_name,
        available_embeddings[best_name],
        df,
        labels,
        rows,
    )


if __name__ == "__main__":
    main()
