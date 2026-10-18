
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "data/code_final_v2.parquet"
LABEL_PATH = "labels_all.csv"
EMBEDDING_PATH = "embeddings_all-mpnet-base-v2.npy"

CLASSES = ["bug", "feature", "other"]
FIXED_THRESHOLD = 0.15

# Inner-CV thresholds to consider.
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def make_features(train_df, test_df, train_embeddings, test_embeddings):
    """Fit text features on training data only."""

    train_text = train_df["prompt"].fillna("").astype(str).tolist()
    test_text = test_df["prompt"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    x_train_tfidf = vectorizer.fit_transform(train_text)
    x_test_tfidf = vectorizer.transform(test_text)

    # Standardize embeddings using training rows only.
    scaler = StandardScaler()
    x_train_emb = scaler.fit_transform(train_embeddings)
    x_test_emb = scaler.transform(test_embeddings)

    # Keep the feature matrices sparse.
    from scipy.sparse import csr_matrix, hstack

    x_train = hstack([
        x_train_tfidf,
        csr_matrix(x_train_emb),
    ]).tocsr()

    x_test = hstack([
        x_test_tfidf,
        csr_matrix(x_test_emb),
    ]).tocsr()

    return x_train, x_test


def threshold_predictions(probabilities, threshold):
    """Use the other class only when its score reaches the threshold."""

    other_index = CLASSES.index("other")
    bug_feature_indices = [
        CLASSES.index("bug"),
        CLASSES.index("feature"),
    ]

    predictions = []

    for row in probabilities:
        if row[other_index] >= threshold:
            predictions.append("other")
        else:
            best_index = bug_feature_indices[
                np.argmax(row[bug_feature_indices])
            ]
            predictions.append(CLASSES[best_index])

    return np.array(predictions)


def fit_and_predict(
    train_df,
    test_df,
    embeddings_by_id,
):
    train_embeddings = np.stack([
        embeddings_by_id[i] for i in train_df["instance_id"]
    ])
    test_embeddings = np.stack([
        embeddings_by_id[i] for i in test_df["instance_id"]
    ])

    x_train, x_test = make_features(
        train_df,
        test_df,
        train_embeddings,
        test_embeddings,
    )

    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )

    model.fit(x_train, train_df["label"])

    raw_probabilities = model.predict_proba(x_test)

    # Ensure columns correspond to bug, feature, other.
    probabilities = np.zeros((len(test_df), len(CLASSES)))

    for source_index, label in enumerate(model.classes_):
        probabilities[:, CLASSES.index(label)] = (
            raw_probabilities[:, source_index]
        )

    return probabilities


def evaluate(name, y_true, y_pred):
    return {
        "strategy": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=CLASSES,
            average="macro", zero_division=0
        ),
        "other_precision": _class_metric(
            y_true, y_pred, "other", "precision"
        ),
        "other_recall": _class_metric(
            y_true, y_pred, "other", "recall"
        ),
        "predicted_other": int(np.sum(y_pred == "other")),
    }


def _class_metric(y_true, y_pred, target, metric):
    from sklearn.metrics import precision_score, recall_score

    if metric == "precision":
        return precision_score(
            y_true, y_pred, labels=[target],
            average="macro", zero_division=0
        )

    return recall_score(
        y_true, y_pred, labels=[target],
        average="macro", zero_division=0
    )


def main():
    print("Loading dataset and labels...")

    data = pd.read_parquet(DATA_PATH)
    labels = pd.read_csv(LABEL_PATH)

    # Random labels are the evaluation population.
    random_labels = labels[labels["source"] == "random"].copy()

    # Extra uncertain labels may be used for training only.
    extra_labels = labels[labels["source"] != "random"].copy()

    random_labels = random_labels[
        random_labels["label"].isin(CLASSES)
    ].reset_index(drop=True)

    extra_labels = extra_labels[
        extra_labels["label"].isin(CLASSES)
    ].reset_index(drop=True)

    # Only retain IDs available in the canonical dataset.
    valid_ids = set(data["instance_id"])
    random_labels = random_labels[
        random_labels["instance_id"].isin(valid_ids)
    ].reset_index(drop=True)
    extra_labels = extra_labels[
        extra_labels["instance_id"].isin(valid_ids)
    ].reset_index(drop=True)

    # Merge prompt text without relying on row order.
    data = data.drop_duplicates("instance_id").set_index("instance_id")
    random_labels["prompt"] = random_labels["instance_id"].map(
        data["prompt"]
    )
    extra_labels["prompt"] = extra_labels["instance_id"].map(
        data["prompt"]
    )

    embeddings = np.load(EMBEDDING_PATH)

    # The embedding array is assumed to follow the canonical parquet row
    # order. Verify this assumption before proceeding.
    canonical_ids = data.index.tolist()

    if len(embeddings) != len(canonical_ids):
        raise ValueError(
            f"Embedding rows ({len(embeddings)}) do not match "
            f"unique dataset IDs ({len(canonical_ids)})."
        )

    embeddings_by_id = {
        instance_id: embeddings[i]
        for i, instance_id in enumerate(canonical_ids)
    }

    random_labels = random_labels.dropna(subset=["prompt"])
    extra_labels = extra_labels.dropna(subset=["prompt"])

    print(f"Random labeled examples: {len(random_labels)}")
    print(f"Extra training examples: {len(extra_labels)}")
    print("\nRandom-label distribution:")
    print(random_labels["label"].value_counts())

    # Every class must appear in each fold.
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    all_true = []
    all_baseline = []
    all_fixed = []
    all_tuned = []
    selected_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(random_labels, random_labels["label"]), start=1
    ):
        print(f"\nOuter fold {fold}/5")

        outer_train = random_labels.iloc[train_idx].reset_index(drop=True)
        outer_test = random_labels.iloc[test_idx].reset_index(drop=True)

        # Threshold selection uses only inner-CV predictions from
        # the outer training portion. Outer test labels remain unseen.
        inner_cv = StratifiedKFold(
            n_splits=4, shuffle=True, random_state=100 + fold
        )

        inner_true = []
        inner_probabilities = []

        for inner_train_idx, inner_val_idx in inner_cv.split(
            outer_train, outer_train["label"]
        ):
            inner_train = outer_train.iloc[
                inner_train_idx
            ].reset_index(drop=True)
            inner_val = outer_train.iloc[
                inner_val_idx
            ].reset_index(drop=True)

            # Additional uncertain labels are training-only.
            training_data = pd.concat(
                [inner_train, extra_labels],
                ignore_index=True,
            )

            # Avoid duplicate IDs if an extra example overlaps
            # the random training examples.
            training_data = training_data.drop_duplicates(
                "instance_id", keep="first"
            ).reset_index(drop=True)

            probs = fit_and_predict(
                training_data,
                inner_val,
                embeddings_by_id,
            )

            inner_true.extend(inner_val["label"].tolist())
            inner_probabilities.extend(probs)

        inner_true = np.array(inner_true)
        inner_probabilities = np.array(inner_probabilities)

        # Select threshold only from inner-CV results.
        threshold_scores = []

        for threshold in THRESHOLDS:
            pred = threshold_predictions(
                inner_probabilities, threshold
            )

            score = f1_score(
                inner_true,
                pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )

            threshold_scores.append((score, threshold))

        # Prefer the higher threshold when macro F1 ties.
        best_score, best_threshold = max(
            threshold_scores, key=lambda item: (item[0], item[1])
        )

        selected_thresholds.append(best_threshold)
        print(
            f"Selected threshold: {best_threshold:.2f} "
            f"(inner macro F1={best_score:.3f})"
        )

        # Train on the full outer training fold plus extra labels.
        full_training_data = pd.concat(
            [outer_train, extra_labels],
            ignore_index=True,
        ).drop_duplicates("instance_id", keep="first")
        full_training_data = full_training_data.reset_index(drop=True)

        outer_probabilities = fit_and_predict(
            full_training_data,
            outer_test,
            embeddings_by_id,
        )

        baseline_pred = np.array([
            CLASSES[i]
            for i in np.argmax(outer_probabilities, axis=1)
        ])

        fixed_pred = threshold_predictions(
            outer_probabilities, FIXED_THRESHOLD
        )

        tuned_pred = threshold_predictions(
            outer_probabilities, best_threshold
        )

        all_true.extend(outer_test["label"].tolist())
        all_baseline.extend(baseline_pred.tolist())
        all_fixed.extend(fixed_pred.tolist())
        all_tuned.extend(tuned_pred.tolist())

    y_true = np.array(all_true)

    results = []
    for name, predictions in [
        ("baseline_argmax", np.array(all_baseline)),
        ("fixed_threshold_0.15", np.array(all_fixed)),
        ("inner_cv_selected_threshold", np.array(all_tuned)),
    ]:
        results.append(evaluate(name, y_true, predictions))

        print(f"\n{'=' * 65}")
        print(name)
        print(classification_report(
            y_true,
            predictions,
            labels=CLASSES,
            zero_division=0,
        ))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(confusion_matrix(
            y_true, predictions, labels=CLASSES
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv("nested_cv_results.csv", index=False)

    pd.DataFrame({
        "outer_fold": np.arange(1, len(selected_thresholds) + 1),
        "selected_threshold": selected_thresholds,
    }).to_csv("nested_cv_thresholds.csv", index=False)

    print("\nSUMMARY")
    print(results_df.to_string(index=False))
    print("\nSelected thresholds by fold:", selected_thresholds)
    print("\nSaved nested_cv_results.csv")
    print("Saved nested_cv_thresholds.csv")


if __name__ == "__main__":
    main()
