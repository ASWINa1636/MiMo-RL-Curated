# from datasets import load_dataset

# ds = load_dataset("XiaomiMiMo/MiMo-V2.6-RL-oss", "code",
#                   split="train", streaming=True)

# for i, row in enumerate(ds):
#     print(row.keys())
#     print(row["prompt"][0]["content"][:200])
#     if i == 2:
#         break

# from huggingface_hub import list_repo_files

# files = list_repo_files("XiaomiMiMo/MiMo-V2.6-RL-oss", repo_type="dataset")
# for f in files:
#     print(f)

# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="XiaomiMiMo/MiMo-V2.6-RL-oss",
#     repo_type="dataset",
#     allow_patterns=["code.parquet"],      # change if the printed names look different
#     local_dir="data",
# )

# import pandas as pd, glob

# paths = glob.glob("data/code.parquet")
# df = pd.concat([pd.read_parquet(p) for p in paths])
# print(df.shape)       # about 2,700 rows
# print(df.columns)

import pandas as pd
df = pd.read_parquet("data/code_embedded.parquet")


names = {
    0: "Auth & tokens", 1: "Go libraries", 2: "Kubernetes", 3: "CLI & config",
    4: "Paths & metadata", 5: "Text formatting", 6: "Python tooling",
    7: "Template-style prompts", 8: "Git & releases", 9: "Python API features",
    10: "Home Assistant", 11: "Frontend", 12: "Bytes & JSON (Go)",
    13: "Networking", 14: "JS / React / Jest", 15: "numpy & pandas",
    16: "SQL tools", 17: "HTTP & APIs", 18: "Jobs & workflows",
    19: "Queries & metrics",
}
df["cluster_name"] = df["cluster_id"].map(names)
df.to_parquet("code_final.parquet", index=False)