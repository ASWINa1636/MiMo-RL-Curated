"""Recompute the patch-based columns WITHOUT the harness files.

The original environments add three helper files to every test patch:
  mimo_test_command.sh (all tasks), mimo_build_env.tar.gz.b64 (about 31%), test_commands.json (about 22%).
They are not tests, so they must not count as test files, patch lines, language or difficulty.

Run from your project folder:   python fix_harness_columns.py
Fixes code_final_v4.parquet and data/code_final_v2.parquet (both, if they exist). Safe to run twice.
Pipeline position: after finalize.py / rename_clusters.py, before make_lite.py and export_release.py.
"""
import json
import os
import re
import numpy as np
import pandas as pd

HARNESS = {"mimo_test_command.sh", "mimo_build_env.tar.gz.b64", "test_commands.json"}
EXT = {".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
       ".tsx": "TypeScript", ".go": "Go", ".java": "Java", ".rs": "Rust", ".cpp": "C++", ".cc": "C++", ".c": "C",
       ".h": "C", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".kt": "Kotlin", ".scala": "Scala", ".jl": "Julia",
       ".sh": "Shell"}


def per_file(patch):
    """{file: (added, removed)} for every file in a unified diff"""
    out, cur = {}, None
    for line in patch.splitlines():
        m = re.match(r"diff --git a/(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = [0, 0]
        elif cur and line.startswith("+") and not line.startswith("+++"):
            out[cur][0] += 1
        elif cur and line.startswith("-") and not line.startswith("---"):
            out[cur][1] += 1
    return out


def language(files):
    for f in files:
        for ext, lang in EXT.items():
            if f.endswith(ext):
                return lang
    return "Other"


def fix(df):
    stats = df["test_patch"].apply(per_file)
    real = stats.apply(lambda d: {k: v for k, v in d.items() if k.split("/")[-1] not in HARNESS})
    empty = int((real.apply(len) == 0).sum())
    real = [r if r else s for r, s in zip(real, stats)]          # no real file at all: keep what exists
    df["test_files"] = [json.dumps(list(r)) for r in real]
    df["n_test_files"] = [len(r) for r in real]
    df["patch_added"] = [sum(v[0] for v in r.values()) for r in real]
    df["patch_removed"] = [sum(v[1] for v in r.values()) for r in real]
    df["patch_lines"] = df["patch_added"] + df["patch_removed"]
    df["prog_language"] = [language(list(r)) for r in real]
    pct = lambda s: s.rank(pct=True)
    score = 0.40 * pct(df["patch_lines"]) + 0.30 * pct(df["n_test_files"]) + 0.30 * pct(df["prompt_words"])
    df["difficulty_score"] = score.round(3)
    df["difficulty"] = pd.qcut(df["difficulty_score"].rank(method="first"), 3, labels=["easy", "medium", "hard"]).astype(str)
    return df, empty


for path in ["code_final_v4.parquet", "data/code_final_v2.parquet", "code_final_v2.parquet"]:
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    before = df["difficulty"].astype(str).copy()
    lang_before = df["prog_language"].copy()
    df, empty = fix(df)
    df.to_parquet(path, index=False)
    print(f"\n{path}: {len(df)} rows fixed ({empty} tasks had no non-harness file)")
    print("  difficulty groups:", df["difficulty"].value_counts().to_dict(),
          f"| {(before != df['difficulty']).mean():.0%} of tasks changed group")
    print("  Shell:", int((lang_before == 'Shell').sum()), "->", int((df['prog_language'] == 'Shell').sum()),
          "| Other:", int((df['prog_language'] == 'Other').sum()))
    print("  median test files:", df["n_test_files"].median())
