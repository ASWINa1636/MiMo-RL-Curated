
import json
import pandas as pd
import requests

task_id = "format-code-task-001933"

df = pd.read_parquet("code_final_v4.parquet")
rows = df[df["instance_id"] == task_id]

if rows.empty:
    raise SystemExit(f"Task not found: {task_id}")

issue = str(rows.iloc[0]["prompt"])[:6000]

payload = {
    "model": "qwen2.5-coder:7b",
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["bug", "feature", "other"]
            },
            "reason": {"type": "string"}
        },
        "required": ["label", "reason"],
        "additionalProperties": False
    },
    "system": (
        "You classify GitHub issues. "
        "Do not solve the issue or output configuration code. "
        "Choose exactly one label: bug, feature, or other. "
        "bug = existing behavior is broken. "
        "feature = new functionality is requested. "
        "other = documentation, maintenance, behavior-preserving "
        "refactoring, performance-only work, questions, or unclear tasks. "
        "Return only the requested JSON object."
    ),
    "prompt": (
        "Classify this GitHub issue.\n\n"
        "ISSUE DESCRIPTION:\n"
        f"{issue}\n\n"
        'Your entire response must be JSON like: '
        '{"label":"bug","reason":"Existing behavior is broken."}'
    ),
    "options": {"temperature": 0, "num_ctx": 2048},
    "keep_alive": "10m",
}

response = requests.post(
    "http://localhost:11434/api/generate",
    json=payload,
    timeout=300,
)
response.raise_for_status()

raw = response.json().get("response", "")
print("Raw response:", repr(raw))

try:
    result = json.loads(raw)
    label = result.get("label")
    if label not in {"bug", "feature", "other"}:
        raise ValueError(f"Invalid or missing label: {label!r}")
    print("VALID LABEL:", label)
    print("REASON:", result.get("reason", ""))
except (json.JSONDecodeError, ValueError) as exc:
    print("INVALID MODEL RESPONSE:", exc)
