from __future__ import annotations

import math
import os
import traceback
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"
SHORT_TEXT_TOKEN_THRESHOLD = int(os.getenv("SHORT_TEXT_TOKEN_THRESHOLD", "3"))
OFFICIAL_MODE_WEIGHTS = [0.4, 0.2, 0.4]
SHORT_TEXT_CONTEXT_VIEWS = [
    ("Bu haber {text} ile ilgilidir.", "Bu haber bir konu ile ilgilidir."),
    ("Bu haberin ana konusu {text}.", "Bu haberin ana konusu bir konudur."),
    ("Bu icerikte one cikan kavram {text}.", "Bu icerikte one cikan kavram bir konudur."),
    ("Gundemdeki baslik: {text}.", "Gundemdeki baslik: bir konudur."),
]
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


def token_count(text: str) -> int:
    return len([segment for segment in text.strip().split() if segment])


def subtract_vectors(left: list[float], right: list[float]) -> list[float]:
    return [float(left[index]) - float(right[index]) for index in range(len(left))]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimension = len(vectors[0])
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]


def top_sparse_tokens(model: Any, lexical_weights: Any, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(lexical_weights, dict) or not lexical_weights:
        return []

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return []

    tokens = []
    sorted_items = sorted(lexical_weights.items(), key=lambda item: item[1], reverse=True)[:limit]
    for token_id, weight in sorted_items:
        try:
            token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        except Exception:
            token = str(token_id)
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
    sparse_vectors = encoded.get("lexical_weights") or [{} for _ in texts]
    try:
        sparse_preview = [top_sparse_tokens(model, weights) for weights in sparse_vectors]
    except Exception:
        sparse_preview = [[] for _ in texts]
    return {
        "dense_vectors": dense_vectors,
        "sparse_preview": sparse_preview,
        "dimension": len(dense_vectors[0]) if dense_vectors else 0,
    }


def build_short_text_context_vectors(texts: list[str]) -> dict[str, list[float]]:
    short_texts = [
        text
        for text in dict.fromkeys(texts)
        if token_count(text) <= SHORT_TEXT_TOKEN_THRESHOLD
    ]
    if not short_texts:
        return {}

    model = get_model()
    prompt_texts: list[str] = []
    prompt_entries: list[tuple[str, int, str | None]] = []
    for view_index, (template, neutral) in enumerate(SHORT_TEXT_CONTEXT_VIEWS):
        prompt_texts.append(neutral)
        prompt_entries.append(("neutral", view_index, None))
        for text in short_texts:
            prompt_texts.append(template.format(text=text))
            prompt_entries.append(("text", view_index, text))

    encoded = model.encode(
        prompt_texts,
        batch_size=min(len(prompt_texts), 16),
        max_length=256,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vectors = [
        [float(value) for value in vector]
        for vector in encoded["dense_vecs"]
    ]

    neutral_vectors: dict[int, list[float]] = {}
    for entry, vector in zip(prompt_entries, dense_vectors):
        entry_type, view_index, _ = entry
        if entry_type == "neutral":
            neutral_vectors[view_index] = vector

    contextual_vectors: dict[str, list[list[float]]] = {text: [] for text in short_texts}
    for entry, vector in zip(prompt_entries, dense_vectors):
        entry_type, view_index, text = entry
        if entry_type != "text" or text is None:
            continue
        contextual_vectors[text].append(subtract_vectors(vector, neutral_vectors[view_index]))

    return {
        text: mean_vector(vectors)
        for text, vectors in contextual_vectors.items()
        if vectors
    }


def build_pairwise_scores(texts: list[str]) -> list[dict[str, Any]]:
    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) < 2:
        return []

    model = get_model()
    sentence_pairs: list[list[str]] = []
    for left_index, left_text in enumerate(unique_texts):
        for right_text in unique_texts[left_index + 1:]:
            sentence_pairs.append([left_text, right_text])

    official_scores = model.compute_score(
        sentence_pairs,
        max_passage_length=256,
        weights_for_different_modes=OFFICIAL_MODE_WEIGHTS,
    )
    short_text_vectors = build_short_text_context_vectors(unique_texts)

    pairwise: list[dict[str, Any]] = []
    for index, (left_text, right_text) in enumerate(sentence_pairs):
        dense_score = float(official_scores["dense"][index])
        sparse_score = float(official_scores["sparse"][index])
        colbert_score = float(official_scores["colbert"][index])
        official_hybrid_score = float(official_scores["colbert+sparse+dense"][index])

        short_text_context_score: float | None = None
        score_mode = "official_hybrid"
        final_score = official_hybrid_score
        if left_text in short_text_vectors and right_text in short_text_vectors:
            short_text_context_score = cosine_similarity(
                short_text_vectors[left_text],
                short_text_vectors[right_text],
            )
            final_score = short_text_context_score
            score_mode = "short_text_context"

        pairwise.append(
            {
                "left": left_text,
                "right": right_text,
                "dense_score": round(dense_score, 6),
                "sparse_score": round(sparse_score, 6),
                "colbert_score": round(colbert_score, 6),
                "official_hybrid_score": round(official_hybrid_score, 6),
                "short_text_context_score": (
                    round(short_text_context_score, 6)
                    if short_text_context_score is not None
                    else None
                ),
                "score_mode": score_mode,
                "score": round(final_score, 6),
            }
        )

    return sorted(pairwise, key=lambda item: item["score"], reverse=True)


def build_clusters(texts: list[str], pairwise: list[dict[str, Any]]) -> dict[str, Any]:
    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) < 2:
        return {
            "method": "mutual_top_k_final_score",
            "top_k": 0,
            "clusters": [[text] for text in unique_texts],
            "edges": [],
        }

    top_k = 2 if len(unique_texts) > 4 else 1
    neighbors: dict[str, list[tuple[str, float]]] = {text: [] for text in unique_texts}
    for record in pairwise:
        left_text = record["left"]
        right_text = record["right"]
        score = float(record["score"])
        neighbors[left_text].append((right_text, score))
        neighbors[right_text].append((left_text, score))

    for text in unique_texts:
        neighbors[text].sort(key=lambda item: item[1], reverse=True)

    top_neighbors = {
        text: {neighbor for neighbor, _ in neighbors[text][:top_k]}
        for text in unique_texts
    }

    parent = {text: text for text in unique_texts}

    def find(text: str) -> str:
        while parent[text] != text:
            parent[text] = parent[parent[text]]
            text = parent[text]
        return text

    def union(left_text: str, right_text: str) -> None:
        left_root = find(left_text)
        right_root = find(right_text)
        if left_root != right_root:
            parent[right_root] = left_root

    edges: list[dict[str, Any]] = []
    for record in pairwise:
        left_text = record["left"]
        right_text = record["right"]
        score = float(record["score"])
        is_mutual_top_k = (
            right_text in top_neighbors[left_text]
            and left_text in top_neighbors[right_text]
        )
        is_strong_edge = score >= 0.55
        if is_mutual_top_k or is_strong_edge:
            union(left_text, right_text)
            edges.append(
                {
                    "left": left_text,
                    "right": right_text,
                    "score": round(score, 6),
                    "reason": "mutual_top_k" if is_mutual_top_k else "strong_edge",
                }
            )

    grouped: dict[str, list[str]] = {}
    for text in unique_texts:
        grouped.setdefault(find(text), []).append(text)

    clusters = sorted(
        [sorted(items) for items in grouped.values()],
        key=lambda items: (-len(items), items),
    )
    return {
        "method": "mutual_top_k_final_score",
        "top_k": top_k,
        "clusters": clusters,
        "edges": edges,
    }


def build_similarity_report(groups: dict[str, list[str]]) -> dict[str, Any]:
    flat_texts: list[str] = []
    index_to_group: dict[str, str] = {}
    for group_name, items in groups.items():
        for item in items:
            flat_texts.append(item)
            index_to_group[item] = group_name

    encoded = encode_texts(flat_texts)
    pairwise = build_pairwise_scores(flat_texts)
    within_scores: dict[str, list[float]] = {group_name: [] for group_name in groups}
    between_scores: list[float] = []

    for record in pairwise:
        left_group = index_to_group[record["left"]]
        right_group = index_to_group[record["right"]]
        record["left_group"] = left_group
        record["right_group"] = right_group
        if left_group == right_group:
            within_scores[left_group].append(float(record["score"]))
        else:
            between_scores.append(float(record["score"]))

    return {
        "model": MODEL_NAME,
        "dimension": encoded["dimension"],
        "groups": groups,
        "scoring": {
            "short_text_token_threshold": SHORT_TEXT_TOKEN_THRESHOLD,
            "short_text_mode": "debiased_multiview_dense",
            "long_text_mode": "official_bge_m3_colbert_sparse_dense",
            "official_mode_weights": OFFICIAL_MODE_WEIGHTS,
        },
        "pairwise": pairwise,
        "group_averages": {
            "within": {
                group_name: round(sum(scores) / len(scores), 6) if scores else None
                for group_name, scores in within_scores.items()
            },
            "between": round(sum(between_scores) / len(between_scores), 6) if between_scores else None,
        },
        "clusters": build_clusters(flat_texts, pairwise),
        "sparse_preview": {
            flat_texts[index]: encoded["sparse_preview"][index]
            for index in range(len(flat_texts))
        },
    }


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    include_vectors: bool = False


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(min_length=2, max_length=32)


app = FastAPI(title="BGE-M3 Probe", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/probe")
def probe() -> dict[str, Any]:
    try:
        return build_similarity_report(DEFAULT_PROBE_GROUPS)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "trace": traceback.format_exc().splitlines()[-20:],
            },
        )


@app.post("/embed")
def embed(request: EmbedRequest) -> dict[str, Any]:
    try:
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
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "trace": traceback.format_exc().splitlines()[-20:],
            },
        )


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        encoded = encode_texts(request.texts)
        pairwise = build_pairwise_scores(request.texts)
        return {
            "model": MODEL_NAME,
            "dimension": encoded["dimension"],
            "texts": request.texts,
            "scoring": {
                "short_text_token_threshold": SHORT_TEXT_TOKEN_THRESHOLD,
                "short_text_mode": "debiased_multiview_dense",
                "long_text_mode": "official_bge_m3_colbert_sparse_dense",
                "official_mode_weights": OFFICIAL_MODE_WEIGHTS,
            },
            "pairwise": pairwise,
            "clusters": build_clusters(request.texts, pairwise),
            "sparse_preview": {
                request.texts[index]: encoded["sparse_preview"][index]
                for index in range(len(request.texts))
            },
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "trace": traceback.format_exc().splitlines()[-20:],
            },
        )
