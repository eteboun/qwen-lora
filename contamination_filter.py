from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isclose
from pathlib import Path
from score_cache import ScoreRecord, hash_texts


@dataclass
class ContaminationStepLog:
    input_count: int
    passed_count: int
    rejected_count: int


@dataclass
class SimilarityStepLog(ContaminationStepLog):
    threshold: float | None


@dataclass
class EmbeddingStepLog(SimilarityStepLog):
    model_id: str | None = None
    model_revision: str | None = None


@dataclass
class ContaminationLog:
    reference_count: int | None = None
    scoring_hash: str | None = None
    cache_hit: bool | None = None
    selected_indices: list[int] = field(default_factory=list)
    steps: dict[str, ContaminationStepLog] = field(default_factory=dict)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


@dataclass
class FilterResult:
    kept_indices: list[int]
    log: ContaminationLog


def filter_scores(
    record: ScoreRecord,
    training_texts: list[str],
    *,
    reference: list[str] | None = None,
    similarity_threshold: float = 0.45,
    embedding_threshold: float | None = None,
) -> FilterResult:
    """Apply thresholds and return the retained indices and their run log."""
    scores = record.scores
    _validate_selection(
        similarity_threshold, embedding_threshold, "embedding" in scores["matches"],
    )
    if hash_texts(training_texts) != record.training_text_hash:
        raise ValueError("Training prompts do not match the cached scoring inputs.")
    if reference is not None and hash_texts(reference) != record.reference_text_hash:
        raise ValueError("Reference prompts do not match the cached scoring inputs.")

    log = ContaminationLog(
        reference_count=len(reference) if reference is not None else scores.get("reference_count"),
        scoring_hash=record.scoring_hash,
        cache_hit=record.cache_hit,
    )

    kept = _select_indices(
        log, scores, len(training_texts), similarity_threshold, embedding_threshold,
        record.embedding.get("model_id"), record.embedding.get("model_revision"),
    )

    return FilterResult(kept_indices=kept, log=log)

def _validate_selection(similarity_threshold, embedding_threshold, has_embedding):
    if not 0 < similarity_threshold <= 1:
        raise ValueError("The TF-IDF similarity threshold must be in (0, 1].")
    if embedding_threshold is not None:
        if not has_embedding:
            raise ValueError("The score record must contain embedding scores to apply an embedding threshold.")
        if not -1 <= embedding_threshold <= 1:
            raise ValueError("The embedding similarity threshold must be in [-1, 1].")

def _select_indices(
    log: ContaminationLog, scores, training_count, similarity_threshold, embedding_threshold,
    embedding_model_id, embedding_model_revision,
):
    # Exact match
    exact_rejected = set(scores["exact_rejected_indices"])
    exact_kept = [index for index in range(training_count) if index not in exact_rejected]
    log.steps["exact_match"] = ContaminationStepLog(
        input_count=training_count,
        passed_count=len(exact_kept),
        rejected_count=len(exact_rejected),
    )

    # TF-IDF
    tfidf_kept, tfidf_log = _apply_threshold(exact_kept, scores["matches"]["tfidf"], similarity_threshold)
    log.steps["tfidf"] = SimilarityStepLog(**asdict(tfidf_log), threshold=similarity_threshold)

    # Embedding
    embedding_kept = exact_kept
    if "embedding" in scores["matches"]:
        embedding_kept, embedding_log = _apply_threshold(
            exact_kept, scores["matches"]["embedding"], embedding_threshold,
        )
        log.steps["embedding"] = EmbeddingStepLog(
            **asdict(embedding_log), threshold=embedding_threshold,
            model_id=embedding_model_id, model_revision=embedding_model_revision,
        )

    # Combine
    kept_indices = _combine_kept_indices(tfidf_kept, embedding_kept)
    log.steps["combined"] = ContaminationStepLog(
        input_count=len(exact_kept),
        passed_count=len(kept_indices),
        rejected_count=len(exact_kept) - len(kept_indices),
    )

    log.selected_indices = kept_indices.copy()
    return kept_indices

def _combine_kept_indices(tfidf_kept: list[int], embedding_kept: list[int]) -> list[int]:
    embedding_kept_set = set(embedding_kept)
    return [index for index in tfidf_kept if index in embedding_kept_set]

def _apply_threshold(kept_indices, matches, threshold):
    kept = kept_indices
    if threshold is not None and matches:
        rejected = {
            match["training_index"]
            for match in matches
            if match["similarity"] >= threshold
            or isclose(match["similarity"], threshold, rel_tol=1e-12)
        }
        kept = [index for index in kept_indices if index not in rejected]
    return kept, ContaminationStepLog(
        input_count=len(kept_indices),
        passed_count=len(kept),
        rejected_count=len(kept_indices) - len(kept),
    )
