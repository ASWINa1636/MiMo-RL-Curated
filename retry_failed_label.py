import json
import time
import pandas as pd
import requests

path = "ollama_label_suggestions.csv"
task_id = "format-code-task-001933"
model = "qwen2.5-coder:7b"

df = pd.read_csv(path)
mask = df["instance_id"].eq(task_id)

if not mask.any():
    raise SystemExit(f"Task not found: {task_id}")

prompt = """
Classify this GitHub coding issue into exactly one category.
bug: Existing functionality is broken, incorrect, or crashes.
feature: New functionality or an additional capability is requested.
other: Documentation, refactoring without behavior changes, maintenance,
performance-only work, general questions, or unclear requests.

Return JSON with exactly these fields:
{"label":"bug|feature|other","reason":"brief explanation"}

Issue:
""" + str(df.loc[mask, "reason"].iloc[0])

# The existing CSV only stores the model's explanation, not the original issue.
# Use the original dataset to retrieve the actual issue description.
try:
    import pyarrow.parquet as pq
    tasks = pd.read_parquet("code_final_v4.parquet")
    issue = tasks.loc[
        tasks["instance_id"].eq(task_id), "prompt"
    ].iloc[0]
except Exception as exc:
    raise SystemExit(f"Could not load original issue: {exc}")

for attempt in range(1, 4):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "system": (
                    "You are a careful coding-issue classifier. "
                    "Choose only bug, feature, or other. "
                    "Use the issue description as the primary evidence. "
                    'Return JSON: {"label":"bug|feature|other",'
                    '"reason":"brief explanation"}.'
                ),
                "prompt": f"Task ID: {task_id}\nIssue:\n{issue}",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 2048},
                "keep_alive": "10m",
            },
            timeout=300,
        )
        response.raise_for_status()
        result = json.loads(response.json()["response"])
        label = str(result.get("label", "")).strip().lower()
        reason = str(result.get("reason", "")).strip()

        if label not in {"bug", "feature", "other"}:
            raise ValueError(f"Invalid label: {label!r}")

        df.loc[mask, "suggested_label"] = label
        df.loc[mask, "reason"] = reason
        df.loc[mask, "review_status"] = "pending"
        df.to_csv(path, index=False)

        print(f"Retry succeeded: {task_id} -> {label}")
        print(f"Reason: {reason}")
        print(f"Updated: {path}")
        break

    except Exception as exc:
        print(f"Attempt {attempt}/3 failed: {exc}")
        if attempt < 3:
            time.sleep(2)
else:
    print("All retries failed. The row remains marked as error.")
