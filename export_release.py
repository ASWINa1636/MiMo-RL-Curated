"""Build the folder you upload to Hugging Face.
Run from your project folder AFTER rename_clusters.py and make_lite.py:
    python export_release.py
Creates hf_release/ with README.md (dataset card), data/full.parquet, data/lite.parquet, embeddings/...
Open README.md and read it once before you upload. Update the numbers in the CONFIG block if you re-run anything.
"""
import os
import json
import numpy as np
import pandas as pd

# ---------------- CONFIG: edit here ----------------
HF_NAME = "mimo-rl-curated-code"                  # repository name on Hugging Face
REPO_URL = "<your-github-repo-url>"               # put your GitHub project link here
V4 = "code_final_v4.parquet"
LITE = "lite/code_lite_500.parquet"
EMB = "embeddings_all-mpnet-base-v2.npy"
OUT = "hf_release"
# numbers printed by train_classifier3.py (cross-validated). Update them after any new run.
ACC = dict(overall=87.4, n_test=228, t70_acc=93.9, t70_share=81.1, t80_acc=97.1, t80_share=71.2)
# After a blind human test, set e.g. HUMAN = dict(n=100, agreement=0.86). None = card says "not done yet".
HUMAN = None
# ---------------------------------------------------

COLS = {   # public column -> description (order = column order in the files)
    "instance_id": "Task id from the original dataset (unique key).",
    "orig_index": "Row index in the original dataset's code subset.",
    "prompt": "The task text (a GitHub-issue style description). Unchanged from the original.",
    "test_patch": "Diff of the test files that verify the task. Unchanged from the original.",
    "docker_image": "Image name given in the original dataset. The images themselves are NOT included here.",
    "verifier_timeout_sec": "Verifier timeout from the original dataset (1800 for every task).",
    "prog_language": "Programming language, guessed from the file extensions in test_patch (rule-based).",
    "test_files": "List of test files touched by test_patch.",
    "n_test_files": "Number of test files in test_patch.",
    "patch_lines": "Added plus removed lines in test_patch.",
    "prompt_words": "Words in the prompt.",
    "prompt_chars": "Characters in the prompt.",
    "has_code_block": "Prompt contains a markdown code fence.",
    "has_traceback": "Prompt contains error/traceback words (rough text flag).",
    "has_chinese": "Prompt contains CJK characters (flags Chinese and some Japanese text).",
    "difficulty": "easy / medium / hard: an ESTIMATE in three equal groups (see 'How the columns were made').",
    "difficulty_score": "0-1 score behind the difficulty groups.",
    "issue_type": "Predicted: bug or feature (see limits).",
    "issue_type_confidence": "Classifier probability (0.5-1.0) of the predicted issue_type.",
    "issue_type_confident": "True when issue_type_confidence >= 0.70.",
    "cluster_id": "Topic cluster number (0-19).",
    "cluster_name": "Short human-written name for the cluster. Names are approximate.",
    "near_duplicate": "True when the closest other task has similarity above 0.90.",
    "nearest_instance_id": "Closest other task (by all-MiniLM-L6-v2 similarity).",
    "nearest_similarity": "Cosine similarity to that closest task.",
}


def prepare(df):
    out = df.rename(columns={"timeout_sec": "verifier_timeout_sec"}).copy()
    out["difficulty"] = out["difficulty"].astype(str)
    out["test_files"] = out["test_files"].apply(json.loads)
    out["nearest_similarity"] = out["nearest_similarity"].astype("float64").round(3)
    return out[list(COLS)].reset_index(drop=True)


full = prepare(pd.read_parquet(V4))
lite = prepare(pd.read_parquet(LITE))

# ---- checks: stop if anything is off ----
assert full["instance_id"].is_unique and lite["instance_id"].is_unique, "duplicate ids"
assert set(lite["instance_id"]) <= set(full["instance_id"]), "lite has ids that are not in full"
assert full.drop(columns=["test_files"]).notna().all().all(), "empty values in full"
assert set(full["issue_type"]) == {"bug", "feature"}, set(full["issue_type"])
emb = np.load(EMB)
assert len(emb) == len(pd.read_parquet(V4)), "embeddings do not match the data"

os.makedirs(f"{OUT}/data", exist_ok=True)
os.makedirs(f"{OUT}/embeddings", exist_ok=True)
full.to_parquet(f"{OUT}/data/full.parquet", index=False)
lite.to_parquet(f"{OUT}/data/lite.parquet", index=False)
pd.DataFrame({"instance_id": pd.read_parquet(V4)["instance_id"].values,
              "embedding": list(emb.astype("float32"))}).to_parquet(f"{OUT}/embeddings/mpnet_embeddings.parquet", index=False)

# ---- numbers for the card (computed from the data, not typed by hand) ----
n, nl = len(full), len(lite)
share = lambda s: f"{(s.mean() * 100):.0f}%"
col_table = "\n".join(f"| `{c}` | {d} |" for c, d in COLS.items())
clusters = full.groupby("cluster_name").size().sort_values(ascending=False)
cluster_table = "\n".join(f"| {c} | {k} |" for c, k in clusters.items())
lang = full["prog_language"].value_counts()
lang_line = ", ".join(f"{k} {v}" for k, v in lang.items())

if HUMAN:
    human = (f"A blind human check of {HUMAN['n']} random tasks agreed with the predicted issue_type on "
             f"{HUMAN['agreement'] * 100:.0f}% of them.")
else:
    human = ("**No independent human validation has been done yet.** The accuracy numbers above measure agreement "
             "with labels that were mostly produced by an LLM, so they may overstate agreement with human judgment.")

card = """---
license: apache-2.0
language:
- en
- zh
pretty_name: MiMo-RL Curated (code)
size_categories:
- 1K<n<10K
task_categories:
- text-classification
tags:
- code
- github-issues
- software-engineering
- data-curation
- reinforcement-learning
source_datasets:
- XiaomiMiMo/MiMo-V2.6-RL-oss
configs:
- config_name: full
  data_files: data/full.parquet
  default: true
- config_name: lite
  data_files: data/lite.parquet
---

# MiMo-RL Curated (code)

A cleaned, flattened and labeled copy of the **`code` subset** of
[XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss),
with extra columns that make it easier to filter, sample and explore.
It is an independent project and is **not affiliated with or endorsed by Xiaomi MiMo**.
All task texts and test patches come from the original dataset, unchanged.

- **full**: @@N@@ tasks. **lite**: @@NL@@ tasks (25 per topic cluster), small enough for a laptop.
- Only the `code` subset is covered (not `webdev`, `music`, `cyber` or `general`).
- License: Apache-2.0, the same as the original dataset. Please also read the original dataset card for any extra terms,
  and note that task texts describe third-party open-source projects.

## What was added

| Added | What it is |
|---|---|
| Flat columns | The original nested `prompt`, `reward_model` and `extra_info` (with a JSON string inside) are unpacked into plain columns. |
| Profile columns | prompt length, code block / traceback flags, Chinese-text flag, test files, patch size. |
| `prog_language` | Language of the test files (rule-based). |
| `difficulty` | An estimated easy / medium / hard group. |
| `issue_type` | Predicted bug or feature, with a confidence score. |
| `cluster_id`, `cluster_name` | 20 automatic topic groups. |
| `near_duplicate` | Flags very similar tasks. |

Removed because they carry no information: `data_source`, `ability`, `agent_name` (the same value in every row),
the empty `reward_model.ground_truth`, and the duplicate `problem_statement` (identical to `prompt`).

## Quick start

```python
from datasets import load_dataset

lite = load_dataset("<your-username>/@@NAME@@", "lite", split="train")
easy_english = lite.filter(lambda r: r["difficulty"] == "easy" and not r["has_chinese"])
sure_bugs = lite.filter(lambda r: r["issue_type"] == "bug" and r["issue_type_confident"])
```

`embeddings/mpnet_embeddings.parquet` holds one `all-mpnet-base-v2` vector per task (key: `instance_id`) for similarity search.

## Columns

| Column | Description |
|---|---|
@@COLS@@

## How the columns were made

- **prog_language**: first file extension in `test_patch` that matches a known language. It is the language of the *tests*;
  a project in one language can have shell-script tests (`Shell` is {} tasks).
- **difficulty**: an *estimate*, not measured model performance. Score = 40% test-patch size + 30% number of test files +
  30% prompt length, each turned into a percentile rank, then cut into three equal groups. "Hard" means "larger than most
  other tasks here". A short task can still be hard.
- **issue_type**: logistic regression on `all-mpnet-base-v2` embeddings + TF-IDF words + a few text flags, trained on about 350 labeled
  tasks (mostly labeled by an LLM, plus some manual labels and corrections). Only two classes are predicted; the roughly 9% of tasks
  that are neither a bug nor a feature (refactors, docs, performance, unclear) are predicted as bug or feature anyway.
- **cluster_id / cluster_name**: KMeans (k = 20) on `all-MiniLM-L6-v2` embeddings. Names were written by reading samples and are
  approximate. Some clusters group by writing style (many prompts start "I want...") as well as by topic.
- **near_duplicate / nearest_***: cosine similarity on `all-MiniLM-L6-v2` embeddings (threshold 0.90). The published embeddings use
  a different model (`all-mpnet-base-v2`), so neighbours found with them can differ.
- **has_chinese**: any CJK character in the prompt.

## Accuracy of `issue_type` (read this before relying on it)

| Measure | Result |
|---|---|
| Bug vs feature, all tasks that were clearly one of the two (n = @@NTEST@@, cross-validated) | **@@ACC@@%** |
| Only predictions with confidence >= 0.70 (@@S70@@% of those tasks) | **@@A70@@%** |
| Only predictions with confidence >= 0.80 (@@S80@@% of those tasks) | **@@A80@@%** |

- Truth for these numbers is the **original** label of each test task, so later label corrections cannot raise the score.
- The test set is small (@@NTEST@@ tasks): expect about plus or minus 4-5 points.
- @@HUMAN@@
- Predictions are less reliable for Chinese-language prompts (average confidence about 0.73 vs 0.88 for the rest).
- For analysis, filter on `issue_type_confident` when you need labels you can trust.

## Other limits

- `difficulty` and `cluster_name` are heuristics, not ground truth.
- Languages are unbalanced: @@LANG@@.
- `lite` is balanced on purpose (25 per cluster), so it is **not a random sample**: it has more Chinese prompts (5.6% vs 3.4%)
  and a lower share of confident labels than the full set.
- The Docker images and verifiers of the original environments are not included.
  To run the tasks, use the original dataset.

## Clusters (full set)

| Cluster | Tasks |
|---|---|
@@CLUSTERS@@

## Credit

Task data: XiaomiMiMo/MiMo-V2.6-RL-oss (Apache-2.0). Added columns and tooling: this project, @@REPO@@.
""".replace("@@N@@", f"{n:,}").replace("@@NL@@", str(nl)).replace("@@NAME@@", HF_NAME) \
   .replace("@@COLS@@", col_table).replace("@@NTEST@@", str(ACC["n_test"])).replace("@@ACC@@", str(ACC["overall"])) \
   .replace("@@S70@@", str(ACC["t70_share"])).replace("@@A70@@", str(ACC["t70_acc"])) \
   .replace("@@S80@@", str(ACC["t80_share"])).replace("@@A80@@", str(ACC["t80_acc"])) \
   .replace("@@HUMAN@@", human).replace("@@LANG@@", lang_line).replace("@@CLUSTERS@@", cluster_table) \
   .replace("@@REPO@@", REPO_URL)
card = card.replace("(`Shell` is {} tasks)", f"(`Shell` is {int(lang.get('Shell', 0))} tasks)")

open(f"{OUT}/README.md", "w", encoding="utf-8").write(card)
for p in ["README.md", "data/full.parquet", "data/lite.parquet", "embeddings/mpnet_embeddings.parquet"]:
    print(f"{OUT}/{p:42s} {os.path.getsize(f'{OUT}/{p}') / 1e6:6.1f} MB")
print(f"\nfull {len(full)} rows, lite {len(lite)} rows, {len(COLS)} columns. Checks passed.")