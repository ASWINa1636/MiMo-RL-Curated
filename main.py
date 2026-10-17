"""MiMo-RL Curated API.

Run from your project folder:   uvicorn main:app --reload
Then open:                      http://127.0.0.1:8000        (dashboard)
                                http://127.0.0.1:8000/docs   (API test page)

Looks for these files (in this folder or in ./data):
  code_final_v4.parquet  (or v3, or v2)    <- the tasks table, newest version wins
  embeddings_all-mpnet-base-v2.npy         <- (or embeddings.npy) one row per task, same order
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


def find(*names):
    for name in names:
        for folder in (".", "data"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"None of these files were found (here or in ./data): {names}")


DATA = find("code_final_v4.parquet", "code_final_v3.parquet", "code_final_v2.parquet")
EMB_FILE = os.environ.get("EMB_FILE") or find("embeddings_all-mpnet-base-v2.npy", "embeddings.npy")

df = pd.read_parquet(DATA).reset_index(drop=True)
emb = np.load(EMB_FILE)
assert len(df) == len(emb), f"{DATA} has {len(df)} rows but {EMB_FILE} has {len(emb)}"
pos = {iid: i for i, iid in enumerate(df["instance_id"])}

# the model that made the embeddings (needed to embed a NEW issue the same way)
MODEL_NAME = {768: "all-mpnet-base-v2", 384: "all-MiniLM-L6-v2"}.get(emb.shape[1])
TEXT_CHARS = 600 if emb.shape[1] == 768 else 1000

df["snippet"] = df["prompt"].str.replace(r"\s+", " ", regex=True).str.slice(0, 220)

HAS_CONF = "issue_type_confidence" in df.columns
HAS_CONFIDENT = "issue_type_confident" in df.columns

LIST_COLS = [c for c in [
    "instance_id", "snippet", "prog_language", "difficulty", "difficulty_score",
    "issue_type", "issue_type_confidence", "issue_type_confident",
    "cluster_name", "prompt_words", "has_chinese", "near_duplicate",
] if c in df.columns]

app = FastAPI(title="MiMo-RL Curated API", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "null"],   # "null" = dashboard.html opened as a file
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def home():
    """The dashboard page (dashboard.html next to this file)."""
    for folder in (".", "data"):
        path = os.path.join(folder, "dashboard.html")
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(404, "dashboard.html was not found next to main.py")


def clean(records):
    """numpy / NaN values -> plain python values that JSON can carry"""
    out = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            row[k] = v
        out.append(row)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(df), "data_file": DATA,
            "embeddings_file": EMB_FILE, "embedding_model": MODEL_NAME}


@app.get("/stats")
def stats():
    def counts(col):
        return df[col].astype(str).value_counts().to_dict() if col in df.columns else {}

    bins = [0, 100, 200, 300, 400, 600, 1000, 100000]
    labels = ["<100", "100-199", "200-299", "300-399", "400-599", "600-999", "1000+"]
    hist = pd.cut(df["prompt_words"], bins=bins, labels=labels, right=False).value_counts().reindex(labels)

    out = {
        "total": len(df),
        "difficulty": counts("difficulty"),
        "issue_type": counts("issue_type"),
        "prog_language": counts("prog_language"),
        "cluster_name": counts("cluster_name"),
        "chinese_prompts": int(df["has_chinese"].sum()),
        "near_duplicates": int(df["near_duplicate"].sum()),
        "prompt_words_hist": {k: int(v) for k, v in hist.items()},
        "median_prompt_words": float(df["prompt_words"].median()),
    }
    if HAS_CONFIDENT:
        out["issue_type_confident"] = int(df["issue_type_confident"].sum())
    if HAS_CONF:
        c = df["issue_type_confidence"]
        out["issue_type_confidence_hist"] = {
            "<0.6": int((c < 0.6).sum()),
            "0.6-0.7": int(((c >= 0.6) & (c < 0.7)).sum()),
            "0.7-0.8": int(((c >= 0.7) & (c < 0.8)).sum()),
            "0.8+": int((c >= 0.8).sum()),
        }
    return out


@app.get("/tasks")
def tasks(
    difficulty: Optional[str] = None,
    issue_type: Optional[str] = None,
    language: Optional[str] = None,
    cluster: Optional[str] = None,
    chinese: Optional[bool] = None,
    confident: Optional[bool] = Query(None, description="only confident issue_type predictions"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    q: Optional[str] = Query(None, description="text contained in the prompt"),
    sort_by: str = Query("instance_id",
                         pattern="^(instance_id|difficulty_score|prompt_words|issue_type_confidence)$"),
    desc: bool = False,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    if sort_by == "issue_type_confidence" and not HAS_CONF:
        raise HTTPException(400, "this data file has no issue_type_confidence column")

    m = pd.Series(True, index=df.index)
    if difficulty:
        m &= df["difficulty"].astype(str) == difficulty
    if issue_type:
        m &= df["issue_type"] == issue_type
    if language:
        m &= df["prog_language"] == language
    if cluster:
        m &= df["cluster_name"] == cluster
    if chinese is not None:
        m &= df["has_chinese"] == chinese
    if confident is not None and HAS_CONFIDENT:
        m &= df["issue_type_confident"] == confident
    if min_confidence is not None and HAS_CONF:
        m &= df["issue_type_confidence"] >= min_confidence
    if min_words is not None:
        m &= df["prompt_words"] >= min_words
    if max_words is not None:
        m &= df["prompt_words"] <= max_words
    if q:
        m &= df["prompt"].str.contains(q, case=False, regex=False)

    sub = df[m].sort_values(sort_by, ascending=not desc)
    page = sub.iloc[offset: offset + limit][LIST_COLS]
    return {"total": int(len(sub)), "limit": limit, "offset": offset,
            "items": clean(page.to_dict("records"))}


@app.get("/tasks/{instance_id}")
def task(instance_id: str, include_patch: bool = False):
    if instance_id not in pos:
        raise HTTPException(404, "unknown instance_id")
    row = df.iloc[pos[instance_id]].drop(labels=["snippet"])
    if not include_patch:
        row = row.drop(labels=["test_patch"])
    return clean([row.to_dict()])[0]


def top_similar(vec, k, skip=None):
    sims = emb @ vec
    if skip is not None:
        sims[skip] = -1
    top = np.argsort(-sims)[:k]
    out = df.iloc[top][LIST_COLS].to_dict("records")
    for r, j in zip(out, top):
        r["similarity"] = round(float(sims[j]), 3)
    return clean(out)


@app.get("/similar/{instance_id}")
def similar(instance_id: str, k: int = Query(5, ge=1, le=20)):
    if instance_id not in pos:
        raise HTTPException(404, "unknown instance_id")
    i = pos[instance_id]
    return {"query": instance_id, "results": top_similar(emb[i], k, skip=i)}


class TextIn(BaseModel):
    text: str
    k: int = 5


_model = None


@app.post("/similar-text")
def similar_text(body: TextIn):
    """Paste a NEW issue and get the closest tasks (loads the embedding model on first use)."""
    global _model
    if MODEL_NAME is None:
        raise HTTPException(503, f"unknown embedding size {emb.shape[1]}; cannot embed new text")
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise HTTPException(503, "sentence-transformers is not installed")
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = 256
    vec = _model.encode([body.text[:TEXT_CHARS]], normalize_embeddings=True)[0]
    return {"results": top_similar(vec, max(1, min(body.k, 20)))}