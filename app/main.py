from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"
DEFAULT_PROBE_GROUPS = {
    "earthquake": ["deprem", "sarsıntı", "zelzele"],
    "football": ["futbol", "transfer", "gol"],
}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def top_sparse_tokens(model: Any, lexical_weights: dict[int, float], limit: int = 8) -> list[dict[str, Any]]:
    tokens = []
    tokenizer = model.tokenizer
    sorted_items = sorted(lexical_weights.items(), key=lambda item: item[1], reverse=True)[:limit]
    for token_id, weight in sorted_items:
        token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        tokens.append({"token": token, "weight": round(float(weight), 6)})
    return tokens


@lru_cache(maxsize=1)
def get_model() -> Any:
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)


def encode_texts(texts: list[str]) -> dict[str, Any]:
    model = get_model()
    encoded = model.encode(
        texts,
        batch_size=min(len(texts), 8),
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vectors = [
        [float(value) for value in vector]
        for vector in encoded["dense_vecs"]
    ]
    sparse_vectors = encoded["lexical_weights"]
    sparse_preview = [top_sparse_tokens(model, weights) for weights in sparse_vectors]
    return {
        "dense_vectors": dense_vectors,
        "sparse_preview": sparse_preview,
        "dimension": len(dense_vectors[0]) if dense_vectors else 0,
    }


def build_similarity_report(groups: dict[str, list[str]]) -> dict[str, Any]:
    flat_texts: list[str] = []
    index_to_group: dict[int, str] = {}
    for group_name, items in groups.items():
        for item in items:
            index = len(flat_texts)
            flat_texts.append(item)
            index_to_group[index] = group_name

    encoded = encode_texts(flat_texts)
    dense_vectors = encoded["dense_vectors"]
    similarities: list[dict[str, Any]] = []
    within_scores: dict[str, list[float]] = {group_name: [] for group_name in groups}
    between_scores: list[float] = []

    for left_index, left_text in enumerate(flat_texts):
        for right_index in range(left_index + 1, len(flat_texts)):
            right_text = flat_texts[right_index]
            score = cosine_similarity(dense_vectors[left_index], dense_vectors[right_index])
            left_group = index_to_group[left_index]
            right_group = index_to_group[right_index]
            record = {
                "left": left_text,
                "right": right_text,
                "left_group": left_group,
                "right_group": right_group,
                "score": round(score, 6),
            }
            similarities.append(record)
            if left_group == right_group:
                within_scores[left_group].append(score)
            else:
                between_scores.append(score)

    return {
        "model": MODEL_NAME,
        "dimension": encoded["dimension"],
        "groups": groups,
        "pairwise": sorted(similarities, key=lambda item: item["score"], reverse=True),
        "group_averages": {
            "within": {
                group_name: round(sum(scores) / len(scores), 6) if scores else None
                for group_name, scores in within_scores.items()
            },
            "between": round(sum(between_scores) / len(between_scores), 6) if between_scores else None,
        },
        "sparse_preview": {
            flat_texts[index]: encoded["sparse_preview"][index]
            for index in range(len(flat_texts))
        },
    }


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    include_vectors: bool = False


app = FastAPI(title="BGE-M3 Probe", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/probe")
def probe() -> dict[str, Any]:
    return build_similarity_report(DEFAULT_PROBE_GROUPS)


@app.post("/embed")
def embed(request: EmbedRequest) -> dict[str, Any]:
    encoded = encode_texts(request.texts)
    response = {
        "model": MODEL_NAME,
        "dimension": encoded["dimension"],
        "texts": request.texts,
        "sparse_preview": {
            request.texts[index]: encoded["sparse_preview"][index]
            for index in range(len(request.texts))
        },
    }
    if request.include_vectors:
        response["dense_vectors"] = encoded["dense_vectors"]
    return response
