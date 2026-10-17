# MiMo-RL Curated

**A reproducible data-curation and exploration pipeline for the Code
subset of Xiaomi's MiMo-V2.6-RL-oss dataset.**

[![Hugging Face
Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Dataset-yellow)](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-181717?logo=github)](https://github.com/ASWINa1636/MiMo-RL-Curated)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

MiMo-RL Curated takes the **`code` subset** of the original
MiMo-V2.6-RL-oss dataset and turns its nested task records into a clean,
enriched dataset that is easier to inspect, filter, compare, and use for
software-engineering research.

> This is an independent community project and is not affiliated with or
> endorsed by Xiaomi MiMo. Original task prompts and test patches are
> retained unchanged. See the [original
> dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
> and its dataset card for source terms and context.

## Published dataset

**Hugging Face:**
[Aswina1636/mimo-rl-curated-code](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code)

  ------------------------------------------------------------------------
  Variant                                       Size Purpose
  --------------------- ---------------------------- ---------------------
  `full`                                 2,698 tasks Full curated Code
                                                     subset

  `lite`                                   500 tasks Laptop-friendly
                                                     sample with 25 tasks
                                                     from each of the 20
                                                     topic clusters
  ------------------------------------------------------------------------

The repository contains the pipeline, classifier, dashboard, label
files, and release tooling. Generated datasets, embeddings, and local
experiment artifacts are intentionally excluded from Git.

## What the project adds

-   **Flattened records:** unpack nested source fields into
    easier-to-use columns.
-   **Task profiling:** prompt word/character counts, test-file counts,
    patch size, code-block and traceback indicators, and a CJK-text
    flag.
-   **Programming-language estimate:** rule-based inference from
    test-file extensions.
-   **Difficulty estimate:** an easy/medium/hard grouping based on task
    size, number of test files, and prompt length. This is a heuristic,
    not measured model difficulty.
-   **Bug-versus-feature classification:** an issue-type prediction and
    confidence score.
-   **Topic clustering:** 20 automatically generated topic groups, with
    human-written approximate names.
-   **Near-duplicate signals:** cosine-similarity-based nearest-task
    metadata to help find similar prompts.
-   **Exploration dashboard:** a Streamlit-based interface for
    inspecting and exploring the curated tasks.

## Pipeline overview

``` text
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

The individual scripts are kept separate so the processing steps can be
inspected and rerun. Review each script's arguments and expected
input/output paths before running the full pipeline.

## Repository structure

  --------------------------------------------------------------------------------
  File                                         Role
  -------------------------------------------- -----------------------------------
  `testing_and_download.py`                    Downloads or tests access to the
                                               source dataset

  `flatten.py`                                 Flattens nested task records

  `difficulty.py`                              Calculates heuristic difficulty
                                               features and groups

  `embed.py`, `embed_v2.py`                    Embedding and semantic-analysis
                                               pipeline variants

  `cluster_report.py`                          Produces cluster summaries/reports

  `rename_clusters.py`                         Applies readable names to topic
                                               clusters

  `label_helper.py`                            Assists with label review

  `train_classifier3.py`                       Trains the bug-versus-feature
                                               classifier

  `finalize.py`                                Finalizes curated data artifacts

  `make_lite.py`                               Creates the 500-task lightweight
                                               variant

  `export_release.py`                          Prepares release artifacts

  `main.py`, `dashboard.html`                  Dashboard application and its HTML
                                               interface

  `labels_all.csv`                             Label data used by the pipeline

  `labels_all_corrected_plus_candidates.csv`   Corrected labels plus candidate
                                               labels for review

  `requirements.txt`                           Python dependencies

  `.gitignore`                                 Excludes generated data and local
                                               artifacts
  --------------------------------------------------------------------------------

Some intermediate artifacts are generated during execution and are not
committed to this repository.

## Dataset fields

The published dataset includes the original task identifiers and task
content, plus added metadata.

  -----------------------------------------------------------------------
  Field                               Meaning
  ----------------------------------- -----------------------------------
  `instance_id`, `orig_index`         Task identifier and original row
                                      index

  `prompt`, `test_patch`              Original task prompt and test patch

  `docker_image`,                     Original environment reference and
  `verifier_timeout_sec`              verifier timeout metadata

  `prog_language`                     Rule-based estimate from test-file
                                      extensions

  `test_files`, `n_test_files`        Test-file list and count

  `patch_lines`                       Added plus removed lines in the
                                      test patch

  `prompt_words`, `prompt_chars`      Prompt length features

  `has_code_block`, `has_traceback`,  Prompt-content indicators
  `has_chinese`                       

  `difficulty`, `difficulty_score`    Heuristic difficulty group and
                                      score

  `issue_type`,                       Predicted bug/feature class,
  `issue_type_confidence`,            confidence, and
  `issue_type_confident`              confidence-threshold flag

  `cluster_id`, `cluster_name`        Topic cluster and approximate
                                      human-readable name

  `near_duplicate`,                   Similarity-based nearest-task
  `nearest_instance_id`,              information
  `nearest_similarity`                
  -----------------------------------------------------------------------

See the [Hugging Face dataset
card](https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code)
for column-generation details, cluster counts, and limitations.

## Quick start: load the published dataset

Python 3.10+ is recommended. Install the Hugging Face Datasets library:

``` bash
python -m pip install -U datasets
```

Load the lightweight dataset:

``` python
from datasets import load_dataset

dataset = load_dataset(
    "Aswina1636/mimo-rl-curated-code",
    "lite",
    split="train",
)

print(dataset)
print(dataset[0])
```

Filter for easy English-language tasks or high-confidence bug
predictions:

``` python
easy_english = dataset.filter(
    lambda row: row["difficulty"] == "easy" and not row["has_chinese"]
)

confident_bugs = dataset.filter(
    lambda row: row["issue_type"] == "bug"
    and row["issue_type_confident"]
)
```

To load the full 2,698-task variant, use the `full` configuration:

``` python
full_dataset = load_dataset(
    "Aswina1636/mimo-rl-curated-code",
    "full",
    split="train",
)
```

## Run the dashboard locally

1.  Clone this repository and enter the project directory.
2.  Create and activate a virtual environment.
3.  Install the listed dependencies.
4.  Start the Streamlit application.

``` bash
git clone https://github.com/ASWINa1636/MiMo-RL-Curated.git
cd MiMo-RL-Curated

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

``` bash
source .venv/bin/activate
```

Install dependencies and launch the dashboard:

``` bash
python -m pip install -r requirements.txt
streamlit run main.py
```

If the dashboard expects generated files that are not included in Git,
run the relevant pipeline steps first. Check the scripts and
`requirements.txt` for the exact dependencies and input paths used by
your local checkout.

## Classification evaluation

The dataset card reports the following cross-validated results for
**bug-versus-feature classification on tasks whose reference labels were
clearly one of those two classes**:

  Evaluation slice                                           Reported accuracy
  -------------------------------------------------------- -------------------
  All clearly bug/feature-labeled test tasks (`n = 228`)                 87.4%
  Predictions with confidence ≥ 0.70 (81.1% coverage)                    93.9%
  Predictions with confidence ≥ 0.80 (71.2% coverage)                    97.1%

**Interpret these numbers carefully:**

-   The evaluation set is small, so results have uncertainty.
-   The reference labels were mostly produced by an LLM; independent
    human validation has not yet been completed.
-   These figures measure agreement with the reference labels, not
    guaranteed agreement with human reviewers or real-world bug/feature
    truth.
-   The classifier forces every task into either `bug` or `feature`.
    Refactors, documentation requests, performance work, and unclear
    tasks may therefore be mislabeled.
-   For downstream analysis, consider filtering on
    `issue_type_confident` and manually reviewing important cases.

The `difficulty` field and cluster names are also estimates, not ground
truth. The `lite` variant is deliberately balanced by topic cluster and
is not a random sample of the full dataset.

## Scope and limitations

-   This release covers only the **Code** subset; it does not include
    the original `webdev`, `music`, `cyber`, or `general` subsets.
-   Original Docker images and verifiers are not bundled in this
    repository or the curated table. Use the original project resources
    if you need to execute tasks in their environments.
-   Programming-language labels are inferred from test-file extensions
    and may differ from the language of the project itself.
-   Similarity scores are signals for review, not proof that two tasks
    are duplicates.
-   Generated Parquet files, embeddings, and intermediate outputs are
    excluded from the Git repository to keep source control lightweight.
    The published dataset is hosted on Hugging Face.

## Reproducibility

For repeatable results:

1.  Record your Python version and installed package versions.
2.  Keep the original dataset version and any source-file changes
    documented.
3.  Run pipeline stages in the intended order and preserve their outputs
    for comparison.
4.  Review label corrections and classifier evaluation splits before
    reporting metrics.
5.  Do not treat heuristic difficulty, cluster labels, or classifier
    predictions as ground truth.

## License and attribution

The curated dataset is published under **Apache-2.0**, matching the
license stated on the original dataset card. Read the [original
MiMo-V2.6-RL-oss dataset
card](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) for
source terms and any additional conditions. Task prompts and test
patches may refer to third-party open-source projects.

This is an independent curation project and is not an official Xiaomi
MiMo release.

## Links

-   **Source code:** https://github.com/ASWINa1636/MiMo-RL-Curated
-   **Curated dataset:**
    https://huggingface.co/datasets/Aswina1636/mimo-rl-curated-code
-   **Original dataset:**
    https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
