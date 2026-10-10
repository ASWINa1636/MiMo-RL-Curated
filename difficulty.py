import pandas as pd

df = pd.read_parquet("data/code_clean.parquet")

def patch_lines(patch):
    added = removed = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed

pl = df["test_patch"].apply(patch_lines)
df["patch_added"]   = pl.apply(lambda x: x[0])
df["patch_removed"] = pl.apply(lambda x: x[1])
df["patch_lines"]   = df["patch_added"] + df["patch_removed"]

df["has_traceback"] = df["prompt"].str.contains("Traceback|Exception|Error:", regex=True)

def pct(s):
    return s.rank(pct=True)

df["difficulty_score"] = (
    0.40 * pct(df["patch_lines"]) +
    0.30 * pct(df["n_test_files"]) +
    0.30 * pct(df["prompt_words"])
).round(3)

df["difficulty"] = pd.qcut(df["difficulty_score"], 3, labels=["easy", "medium", "hard"])

df.to_parquet("data/code_scored.parquet", index=False)
print(df["difficulty"].value_counts())
print(df.groupby("difficulty", observed=True)[["patch_lines","n_test_files","prompt_words"]].median())
