import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("data/code_labeled.parquet")

# the model only reads about 256 tokens, so the first 1000 characters are enough
texts = df["prompt"].str.slice(0, 1000).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)
np.save("embeddings.npy", emb)
print("embeddings:", emb.shape)          # (2698, 384)

# ---- clustering ----
km = KMeans(n_clusters=20, random_state=42, n_init=10)
df["cluster_id"] = km.fit_predict(emb)
print(df["cluster_id"].value_counts().sort_index().to_string())

# ---- near-duplicates ----
sim = emb @ emb.T                         # similarity of every pair
np.fill_diagonal(sim, 0)                  # ignore "similar to itself"
best = sim.argmax(axis=1)
df["nearest_idx"] = best
df["nearest_sim"] = sim[np.arange(len(df)), best].round(3)
df["near_duplicate"] = df["nearest_sim"] > 0.90
print("near-duplicates:", df["near_duplicate"].sum())

df.to_parquet("data/code_embedded.parquet", index=False)

# ---- similar-issue search ----
def similar(i, k=5):
    top = np.argsort(-sim[i])[:k]
    return [(int(j), round(float(sim[i, j]), 3),
             df["prompt"].iloc[j][:100].replace("\n", " ")) for j in top]

print("\nQUERY:", df["prompt"].iloc[0][:100].replace("\n", " "))
for r in similar(0):
    print(r)