import json, re
import pandas as pd

df = pd.read_parquet("data/code.parquet")

def get_prompt(p):
    return p[0]["content"]          # prompt is a list with 1 message

ij = df["extra_info"].apply(lambda x: json.loads(x["instance_json"]))

clean = pd.DataFrame({
    "instance_id":   df["extra_info"].apply(lambda x: x["instance_id"]),
    "index":         df["extra_info"].apply(lambda x: x["index"]),
    "prompt":        df["prompt"].apply(get_prompt),
    "problem_statement": ij.apply(lambda x: x["problem_statement"]),
    "test_patch":    ij.apply(lambda x: x["test_patch"]),
    "docker_image":  ij.apply(lambda x: x["docker_image"]),
    "timeout_sec":   ij.apply(lambda x: x["verifier_timeout_sec"]),
})

# simple profile columns
clean["prompt_chars"] = clean["prompt"].str.len()
clean["prompt_words"] = clean["prompt"].str.split().str.len()
clean["has_code_block"] = clean["prompt"].str.contains("```", regex=False)
clean["has_chinese"] = clean["prompt"].apply(
    lambda t: bool(re.search(r"[\u4e00-\u9fff]", t)))

# files touched by the test patch -> programming language + size
def files_in_patch(patch):
    return re.findall(r"^diff --git a/(\S+)", patch, flags=re.M)

clean["test_files"] = clean["test_patch"].apply(files_in_patch)
clean["n_test_files"] = clean["test_files"].str.len()

EXT = {".py":"Python",".js":"JavaScript",".ts":"TypeScript",".go":"Go",".java":"Java",
       ".rs":"Rust",".cpp":"C++",".c":"C",".rb":"Ruby",".php":"PHP",".cs":"C#",".kt":"Kotlin"}

def lang(files):
    for f in files:
        for e, l in EXT.items():
            if f.endswith(e):
                return l
    return "Other"

clean["prog_language"] = clean["test_files"].apply(lang)
clean["test_files"] = clean["test_files"].apply(json.dumps)

clean.to_parquet("data/code_clean.parquet", index=False)
print(clean.shape)
print(clean["prog_language"].value_counts())
print("Chinese prompts:", clean["has_chinese"].sum())
print("Duplicate prompts:", clean["prompt"].duplicated().sum())