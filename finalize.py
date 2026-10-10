import json
import pandas as pd

df = pd.read_parquet("data/code_final.parquet")

# 1) drop the duplicate column
assert (df["prompt"] == df["problem_statement"]).all()
df = df.drop(columns=["problem_statement"])

# 2) point to the neighbour by instance_id, not by row position
df["nearest_instance_id"] = df["instance_id"].iloc[df["nearest_idx"].values].values
df = df.drop(columns=["nearest_idx"])
df = df.rename(columns={"index": "orig_index", "nearest_sim": "nearest_similarity"})

# 3) better programming-language detection
EXT = {".py":"Python",".js":"JavaScript",".mjs":"JavaScript",".jsx":"JavaScript",
       ".ts":"TypeScript",".tsx":"TypeScript",".go":"Go",".java":"Java",".rs":"Rust",
       ".cpp":"C++",".cc":"C++",".c":"C",".h":"C",".rb":"Ruby",".php":"PHP",
       ".cs":"C#",".kt":"Kotlin",".scala":"Scala",".jl":"Julia",".sh":"Shell"}

def lang(files_json):
    for f in json.loads(files_json):
        for e, l in EXT.items():
            if f.endswith(e):
                return l
    return "Other"

df["prog_language"] = df["test_files"].apply(lang)

# 4) cluster 7 is Chinese-language prompts
df.loc[df["cluster_id"] == 7, "cluster_name"] = "Chinese-language prompts"

df.to_parquet("data/code_final_v2.parquet", index=False)
print(df.shape)
print(df["prog_language"].value_counts())