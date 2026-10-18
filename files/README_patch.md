# README fixes for GitHub (copy each block into README.md)

## 1. Replace the whole "Enriched fields" list with this (it now matches the 25 published columns)

The release contains 25 public columns:

- `instance_id`, `orig_index` - task id and its row index in the original code subset.
- `prompt` - issue/task text (unchanged from the original).
- `test_patch` - test patch supplied with the task (unchanged).
- `docker_image`, `verifier_timeout_sec` - values from the original dataset (the images themselves are not included).
- `test_files`, `n_test_files`, `patch_lines` - real test files in the patch, their count, and changed lines. Three helper files added by the original environments (`mimo_test_command.sh`, `mimo_build_env.tar.gz.b64`, `test_commands.json`) are excluded.
- `prog_language` - language of the real test files (rule-based; `Other` means no known extension). It may differ from the project's main language.
- `prompt_words`, `prompt_chars`, `has_code_block`, `has_traceback`, `has_chinese` - simple text profile.
- `difficulty_score`, `difficulty` - heuristic complexity score and its easy / medium / hard group.
- `issue_type`, `issue_type_confidence`, `issue_type_confident` - predicted bug or feature, the model probability, and a flag for probability >= 0.70.
- `cluster_id`, `cluster_name` - topic cluster number and its short name.
- `near_duplicate`, `nearest_instance_id`, `nearest_similarity` - closest other task and its similarity (flag when above 0.90).

(`patch_added`, `patch_removed` and `issue_type_tier` exist only in the working files, not in the published release.)

## 2. Replace the "Estimated difficulty" paragraph's last sentences with

Tasks are divided into three equal groups: in the current release they contain 900, 899 and 899 tasks. Test-patch size and file count ignore the three helper files that the original environments add to every patch (see the Changelog on the dataset card). "Hard" means relatively larger by these signals within this dataset; it does not mean a model has failed the task.

## 3. Add this section before "Run locally" (exact order, nothing hidden)

## Reproduce the pipeline

Run everything from the project folder. Put the downloaded `code.parquet` there first.

| Step | Command | Writes |
|---|---|---|
| 1 | `python testing_and_download.py` | the original code subset |
| 2 | `python flatten.py` | `code_clean.parquet` |
| 3 | `python difficulty.py` | `code_scored.parquet` (first difficulty version, replaced in step 10) |
| 4 | `python issue_type.py` | `code_labeled.parquet` (old keyword labels; `embed.py` reads this file) |
| 5 | `python embed.py` | `embeddings.npy`, `code_embedded.parquet` (all-MiniLM-L6-v2, clusters, nearest neighbours) |
| 6 | `python cluster_report.py` | `cluster_report.csv` (optional, for reading clusters) |
| 7 | `python name_clusters.py` | `code_final.parquet` |
| 8 | `python finalize.py` | `code_final_v2.parquet` |
| 9 | `python rename_clusters.py` | renames clusters in the v2 file |
| 10 | `python fix_harness_columns.py` | recomputes test files, patch lines, language and difficulty without helper files |
| 11 | `python embed_v2.py all-mpnet-base-v2` | `embeddings_all-mpnet-base-v2.npy` |
| 12 | `python train_classifier3.py` | `code_final_v4.parquet` (needs the two `labels_*.csv` files) |
| 13 | `python make_lite.py` | `lite/code_lite_500.parquet` |
| 14 | `python export_release.py` | `hf_release/` (what is uploaded to Hugging Face) |
| 15 | `uvicorn main:app --reload` | the API and dashboard at http://127.0.0.1:8000 |

Steps 3 and 4 are kept because later steps read their output; the later steps correct their results.

## 4. Optional

- Add GitHub topics (Settings gear next to About): `dataset`, `huggingface`, `data-curation`, `fastapi`, `sentence-transformers`, `software-engineering`, `nlp`.
- Make sure the Mermaid diagram renders (it should show boxes, not code). If it shows code, remove the first line that starts with `%%`.
