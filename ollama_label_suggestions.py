
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "code_final_v4.parquet"
LABEL_PATH = ROOT / "labels_all_corrected_plus_candidates.csv"
OUTPUT_PATH = ROOT / "ollama_label_suggestions.csv"

MODEL = "qwen2.5-coder:7b"
BATCH_SIZE = 20
OLLAMA_URL = "http://localhost:11434/api/generate"

LABELS = {"bug", "feature", "other"}


SYSTEM_PROMPT = """
You classify GitHub coding issues into exactly one category.

bug: Existing functionality is broken, incorrect, crashes, or
behaves differently from its intended behavior.

feature: New functionality, a new capability, or an additional
user-facing option is requested.

other: Documentation-only changes, behavior-preserving refactoring,
maintenance, cleanup, performance-only work, general questions,
or unclear requests.

Classify the issue description, not the implementation task itself.
Do not solve the issue or generate configuration code.
Return only the JSON object required by the output schema.
"""


def classify(prompt, instance_id):
    schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["bug", "feature", "other"],
            },
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    }

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": (
            f"Classify this GitHub issue.\n"
            f"Task ID: {instance_id}\n"
            f"Issue description:\n{str(prompt)[:6000]}\n\n"
            'Return JSON such as {"label":"bug",'
            '"reason":"Existing behavior is broken."}'
        ),
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
        },
        "keep_alive": "10m",
    }

    last_error = None

    for attempt in range(1, 4):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300,
            )
            response.raise_for_status()

            raw = response.json().get("response", "")
            result = json.loads(raw)

            label = str(result.get("label", "")).strip().lower()
            reason = str(result.get("reason", "")).strip()

            if label not in LABELS:
                raise ValueError(
                    f"Invalid or missing label: {label!r}"
                )

            if not reason:
                reason = "No explanation provided by the model."

            return label, reason

        except (
            requests.RequestException,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            print(
                f"  Attempt {attempt}/3 failed for "
                f"{instance_id}: {exc}"
            )

    raise RuntimeError(
        f"Classification failed after 3 attempts: {last_error}"
    )



def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    tasks = pd.read_parquet(DATA_PATH)
    labels = pd.read_csv(LABEL_PATH)

    required = {"instance_id", "prompt"}
    if not required.issubset(tasks.columns):
        raise ValueError(
            f"Dataset missing columns: {required - set(tasks.columns)}"
        )

    if not {"instance_id", "label", "source"}.issubset(labels.columns):
        raise ValueError("Combined labels file has an unexpected schema.")

    if labels["instance_id"].duplicated().any():
        raise ValueError("Duplicate IDs found in combined labels file.")

    labeled_ids = set(labels["instance_id"].astype(str))
    tasks["instance_id"] = tasks["instance_id"].astype(str)

    # Never send already-labeled tasks for new suggestions.
    candidates = tasks[~tasks["instance_id"].isin(labeled_ids)].copy()

    # Resume safely from previous suggestions.

    if OUTPUT_PATH.exists():
        previous = pd.read_csv(OUTPUT_PATH)

        # Only skip rows that have a valid prediction.
        completed = previous[
            previous["review_status"].eq("pending")
            & previous["suggested_label"].isin(LABELS)
        ]

        completed_ids = set(completed["instance_id"].dropna())
    else:
        previous = pd.DataFrame()
        completed_ids = set()


    # Exclude verified labels and previously successful predictions.
    excluded_ids = labeled_ids | {str(x) for x in completed_ids}

    candidates = tasks[
        ~tasks["instance_id"].isin(excluded_ids)
    ].copy()

    # Retry the previously failed task first, if it is still eligible.
    failed_id = "format-code-task-001933"

    failed_mask = candidates["instance_id"].eq(failed_id)
    if failed_mask.any():
        failed_task = candidates.loc[failed_mask]
        remaining_tasks = candidates.loc[~failed_mask]
        candidates = pd.concat(
            [failed_task, remaining_tasks],
            ignore_index=False,
        )

    print(f"Dataset tasks: {len(tasks)}")
    print(f"Existing verified labels: {len(labels)}")
    print(f"Unlabeled tasks available: {len(tasks) - len(labeled_ids)}")
    print(f"Tasks to process this run: {len(candidates)}")
    print(f"Model: {MODEL}")
    print("Suggestions are unverified; review before training.")

    new_rows = []

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        instance_id = row["instance_id"]
        prompt = str(row["prompt"] or "")

        try:
            label, reason = classify(prompt, instance_id)
            record = {
                "instance_id": instance_id,
                "suggested_label": label,
                "reason": reason,
                "review_status": "pending",
                "model": MODEL,
            }
            print(f"[{i}/{len(candidates)}] {instance_id}: {label}")
        except Exception as exc:
            record = {
                "instance_id": instance_id,
                "suggested_label": "",
                "reason": f"ERROR: {exc}",
                "review_status": "error",
                "model": MODEL,
            }
            print(f"[{i}/{len(candidates)}] {instance_id}: ERROR - {exc}")

        new_rows.append(record)

        # Save after every task so interruptions lose little work.
        existing = (
            pd.concat([previous, pd.DataFrame(new_rows)], ignore_index=True)
            if not previous.empty
            else pd.DataFrame(new_rows)
        )
        existing.drop_duplicates(
            subset=["instance_id"], keep="last", inplace=True
        )
        existing.to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.1)

    print(f"\nSaved suggestions to: {OUTPUT_PATH}")
    print("These are predictions, not verified training labels.")


if __name__ == "__main__":
    main()
