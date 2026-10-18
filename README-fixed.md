# MiMo-RL Curated

**A reproducible data-curation and exploration pipeline for the Code subset of Xiaomi's MiMo-V2.6-RL-oss dataset.**

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Dataset-yellow)](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-181717?logo=github)](https://github.com/ASWINa1636/MiMo-RL-Curated)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

MiMo-RL Curated takes the **`code` subset** of the original MiMo-V2.6-RL-oss dataset and turns its nested task records into a clean, enriched dataset that is easier to inspect, filter, compare, and use for software-engineering research.

> This is an independent community project and is not affiliated with or endorsed by Xiaomi MiMo. Original task prompts and test patches are retained unchanged. See the [original dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) and its dataset card for source terms and context.

## Published dataset

**Hugging Face:** [Aswina1636/mimo-rl-curated-code](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code)

This project publishes two dataset configurations:

- **`full` — 2,698 tasks:** the full curated Code subset.
- **`lite` — 500 tasks:** a laptop-friendly sample with 25 tasks from each of the 20 topic clusters.

The repository contains the pipeline, classifier, dashboard, label files, and release tooling. Generated datasets, embeddings, and local experiment artifacts are intentionally excluded from Git.

## What the project adds

- **Flattened records:** unpacks nested source fields into easier-to-use columns.
- **Task profiling:** prompt word and character counts, test-file counts, patch size, code-block and traceback indicators, and a CJK-text flag.
- **Programming-language estimate:** rule-based inference from test-file extensions.
- **Difficulty estimate:** easy, medium, and hard groups based on task size, number of test files, and prompt length. This is a heuristic, not measured model difficulty.
- **Bug-versus-feature classification:** an issue-type prediction and confidence score.
- **Topic clustering:** 20 automatically generated topic groups with human-readable approximate names.
- **Near-duplicate signals:** cosine-similarity-based nearest-task metadata to help find similar prompts.
- **Exploration dashboard:** a Streamlit-based interface for inspecting and exploring curated tasks.

## Pipeline overview

```text
Original MiMo-V2.6-RL-oss Code subset
                  |
                  v
        Download and flatten
                  |
                  v
      Clean and enrich task records
                  |
          +-------+--------+
          |                |
          v                v
   Difficulty scoring   Text embeddings
                           |
                  +--------+---------+
                  |                  |
                  v                  v
             Clustering      Similarity / near-duplicates
                  |
                  v
  Bug-versus-feature classification
                  |
                  v
  Full and lightweight release artifacts
                  |
                  v
   Hugging Face dataset + dashboard
```

The scripts are kept separate so processing steps can be inspected and rerun. Review each script's arguments and expected input/output paths before running the full pipeline.

## Repository structure

- **`testing_and_download.py`** — downloads or tests access to the source dataset.
- **`flatten.py`** — flattens nested task records.
- **`difficulty.py`** — calculates heuristic difficulty features and groups.
- **`embed.py`, `embed_v2.py`** — embedding and semantic-analysis pipeline variants.
- **`cluster_report.py`** — produces cluster summaries and reports.
- **`rename_clusters.py`** — applies readable names to topic clusters.
- **`label_helper.py`** — assists with label review.
- **`train_classifier3.py`** — trains the bug-versus-feature classifier.
- **`finalize.py`** — finalizes curated data artifacts.
- **`make_lite.py`** — creates the 500-task lightweight variant.
- **`export_release.py`** — prepares release artifacts.
- **`main.py`, `dashboard.html`** — dashboard application and its HTML interface.
- **`labels_all.csv`** — label data used by the pipeline.
- **`labels_all_corrected_plus_candidates.csv`** — corrected labels plus candidate labels for review.
- **`requirements.txt`** — Python dependencies.
- **`.gitignore`** — excludes generated data and local artifacts.

Some intermediate artifacts are generated during execution and are not committed to this repository.

## Dataset fields

The published dataset includes original task identifiers and content, plus added metadata. Important fields include:

- **`instance_id`, `orig_index`** — task identifier and original row index.
- **`prompt`, `test_patch`** — original task prompt and test patch.
- **`docker_image`, `verifier_timeout_sec`** — original environment reference and verifier timeout metadata.
- **`prog_language`** — rule-based estimate from test-file extensions.
- **`test_files`, `n_test_files`** — test-file list and count.
- **`patch_lines`** — added plus removed lines in the test patch.
- **`prompt_words`, `prompt_chars`** — prompt length features.
- **`has_code_block`, `has_traceback`, `has_chinese`** — prompt-content indicators.
- **`difficulty`, `difficulty_score`** — heuristic difficulty group and score.
- **`issue_type`, `issue_type_confidence`, `issue_type_confident`** — predicted bug/feature class, confidence, and confidence-threshold flag.
- **`cluster_id`, `cluster_name`** — topic cluster and approximate human-readable name.
- **`near_duplicate`, `nearest_instance_id`, `nearest_similarity`** — similarity-based nearest-task information.

See the [Hugging Face dataset card](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code) for column-generation details, cluster counts, and limitations.

## Quick start: load the published dataset

Python 3.10+ is recommended. Install the Hugging Face Datasets library:

```bash
python -m pip install -U datasets
```

Load the lightweight dataset:

```python
from datasets import load_dataset

dataset = load_dataset(
    "Aswina1636/mimo-rl-curated-code",
    "lite",
    split="train",
)

print(dataset)
print(dataset[0])
```

Filter for easy English-language tasks or bug predictions:

```python
easy_english = dataset.filter(
    lambda row: row["difficulty"] == "easy" and not row["has_chinese"]
)

confident_bugs = dataset.filter(
    lambda row: row["issue_type"] == "bug"
    and row["issue_type_confident"]
)
```

To load the full 2,698-task variant, use the `full` configuration:

```python
full_dataset = load_dataset(
    "Aswina1636/mimo-rl-curated-code",
    "full",
    split="train",
)
```

## Run the dashboard locally

1. Clone this repository and enter the project directory.
2. Create and activate a virtual environment.
3. Install the listed dependencies.
4. Start the Streamlit application.

```bash
git clone https://github.com/ASWINa1636/MiMo-RL-Curated.git
cd MiMo-RL-Curated

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install dependencies and launch the dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run main.py
```

If the dashboard expects generated files that are not included in Git, run the relevant pipeline steps first. Check the scripts and `requirements.txt` for the exact dependencies and input paths used by your checkout.

## Classification evaluation

The dataset card reports these cross-validated results for **bug-versus-feature classification on tasks whose reference labels were clearly one of those two classes**:

- **All clearly bug/feature-labeled test tasks (`n = 228`):** 87.4% reported accuracy.
- **Predictions with confidence ≥ 0.70:** 93.9% accuracy at 81.1% coverage.
- **Predictions with confidence ≥ 0.80:** 97.1% accuracy at 71.2% coverage.

**Interpret these numbers carefully:**

- The evaluation set is small, so results have uncertainty.
- The reference labels were mostly produced by an LLM; independent human validation has not yet been completed.
- These figures measure agreement with the reference labels, not guaranteed agreement with human reviewers or real-world bug/feature truth.
- The classifier assigns every task to either `bug` or `feature`. Refactors, documentation requests, performance work, and unclear tasks may therefore be mislabeled.
- For downstream analysis, consider filtering on `issue_type_confident` and manually reviewing important cases.

The `difficulty` field and cluster names are also estimates, not ground truth. The `lite` variant is deliberately balanced by topic cluster and is not a random sample of the full dataset.

## Scope and limitations

- This release covers only the **Code** subset; it does not include the original `webdev`, `music`, `cyber`, or `general` subsets.
- Original Docker images and verifiers are not bundled in this repository or the curated table. Use the original project resources if you need to execute tasks in their environments.
- Programming-language labels are inferred from test-file extensions and may differ from the language of the project itself.
- Similarity scores are signals for review, not proof that two tasks are duplicates.
- Generated Parquet files, embeddings, and intermediate outputs are excluded from Git to keep source control lightweight. The published dataset is hosted on Hugging Face.

## Reproducibility

For repeatable results:

1. Record your Python version and installed package versions.
2. Keep the original dataset version and any source-file changes documented.
3. Run pipeline stages in the intended order and preserve their outputs for comparison.
4. Review label corrections and classifier evaluation splits before reporting metrics.
5. Do not treat heuristic difficulty, cluster labels, or classifier predictions as ground truth.

## License and attribution

The curated dataset is published under **Apache-2.0**, matching the license stated on the original dataset card. Read the [original MiMo-V2.6-RL-oss dataset card](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) for source terms and any additional conditions. Task prompts and test patches may refer to third-party open-source projects.

This is an independent curation project and is not an official Xiaomi MiMo release.

## Links

- **Source code:** https://github.com/ASWINa1636/MiMo-RL-Curated
- **Curated dataset:** https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code
- **Original dataset:** https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
