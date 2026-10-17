"""Make embeddings with a different (stronger) model.
Usage:  python embed_v2.py all-mpnet-base-v2
        python embed_v2.py BAAI/bge-base-en-v1.5
Writes embeddings_<model>.npy  (same row order as code_final_v2.parquet)
"""
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

name = sys.argv[1] if len(sys.argv) > 1 else "all-mpnet-base-v2"
df = pd.read_parquet("data/code_final_v2.parquet")
texts = df["prompt"].str.slice(0, 600).tolist()

model = SentenceTransformer(name)
model.max_seq_length = 256

chunks = []
for i in range(0, len(texts), 128):          # chunks keep RAM low
    chunks.append(model.encode(texts[i:i + 128], batch_size=16, normalize_embeddings=True))
    print(i, "of", len(texts))
emb = np.vstack(chunks)

out = "embeddings_" + name.replace("/", "_") + ".npy"
np.save(out, emb)
print("saved", out, emb.shape)
