from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from random import Random
from time import perf_counter
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
    score_generation_elapsed_seconds: float | None = None
    selected_indices: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    steps: dict[str, ContaminationStepLog] = field(default_factory=dict)
    max_samples: int | None = None
    random_seed: int | None = None
    training_dataset_metadata: dict[str, object] | None = None
    reference_dataset_metadata: dict[str, object] | None = None

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


class ContaminationFilter:
    def __init__(self):
        self.contamination_log = ContaminationLog()

    def filter_scores(
        self,
        record: ScoreRecord,
        training_texts: list[str],
        *,
        reference: list[str] | None = None,
        similarity_threshold: float = 0.45,
        embedding_threshold: float | None = None,
        max_samples: int | None = None,
        random_seed: int | None = None,
        notify_on_steps: bool = False,
        generate_report: bool = False,
        training_dataset_metadata: dict[str, object] | None = None,
        reference_dataset_metadata: dict[str, object] | None = None,
    ) -> list[int]:
        """Apply thresholds and sample original training indices from precomputed scores."""
        started = perf_counter()
        scores = record.scores
        self._validate_selection(
            similarity_threshold, embedding_threshold, max_samples, "embedding" in scores["matches"],
        )
        if hash_texts(training_texts) != record.training_text_hash:
            raise ValueError("Training prompts do not match the cached scoring inputs.")
        if reference is not None and hash_texts(reference) != record.reference_text_hash:
            raise ValueError("Reference prompts do not match the cached scoring inputs.")

        self.contamination_log = ContaminationLog(
            reference_count=len(reference) if reference is not None else scores.get("reference_count"),
            scoring_hash=record.scoring_hash,
            cache_hit=record.cache_hit,
            score_generation_elapsed_seconds=None if record.cache_hit else scores["elapsed_seconds"],
            max_samples=max_samples,
            random_seed=random_seed,
            training_dataset_metadata=deepcopy(
                record.dataset_metadata.get("training") if training_dataset_metadata is None else training_dataset_metadata,
            ),
            reference_dataset_metadata=deepcopy(
                record.dataset_metadata.get("reference") if reference_dataset_metadata is None else reference_dataset_metadata,
            ),
        )

        kept = self._select_indices(
            scores, len(training_texts), similarity_threshold, embedding_threshold,
            record.embedding.get("model_id"), record.embedding.get("model_revision"),
            max_samples, random_seed, notify_on_steps,
        )
        self.contamination_log.elapsed_seconds = perf_counter() - started

        if generate_report:
            from report import write_contamination_report

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = Path(__file__).resolve().parent / "contamination_runs" / timestamp
            run_dir.mkdir(parents=True)
            self.contamination_log.write_json(run_dir / "contamination.json")
            write_contamination_report(run_dir, training_texts, reference, scores=scores)
        return kept

    @staticmethod
    def _validate_selection(similarity_threshold, embedding_threshold, max_samples, has_embedding):
        if not 0 < similarity_threshold <= 1:
            raise ValueError("The TF-IDF similarity threshold must be in (0, 1].")
        if embedding_threshold is not None:
            if not has_embedding:
                raise ValueError("The score record must contain embedding scores to apply an embedding threshold.")
            if not -1 <= embedding_threshold <= 1:
                raise ValueError("The embedding similarity threshold must be in [-1, 1].")
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be nonnegative.")

    def _select_indices(
        self, scores, training_count, similarity_threshold, embedding_threshold,
        embedding_model_id, embedding_model_revision, max_samples, random_seed, notify_on_steps,
    ):
        # Exact match
        exact_rejected = set(scores["exact_rejected_indices"])
        exact_kept = [index for index in range(training_count) if index not in exact_rejected]
        self.contamination_log.steps["exact_match"] = ContaminationStepLog(
            input_count=training_count,
            passed_count=len(exact_kept),
            rejected_count=len(exact_rejected),
        )

        # TF-IDF
        if notify_on_steps:
            print(f"Applying TF-IDF threshold {similarity_threshold:g}.", flush=True)
        tfidf_kept, tfidf_log = self._apply_threshold(exact_kept, scores["matches"]["tfidf"], similarity_threshold)
        self.contamination_log.steps["tfidf"] = SimilarityStepLog(**asdict(tfidf_log), threshold=similarity_threshold)

        # Embedding
        embedding_kept = exact_kept
        if "embedding" in scores["matches"]:
            if notify_on_steps and embedding_threshold is not None:
                print(f"Applying embedding threshold {embedding_threshold:g}.", flush=True)
            embedding_kept, embedding_log = self._apply_threshold(
                exact_kept, scores["matches"]["embedding"], embedding_threshold,
            )
            self.contamination_log.steps["embedding"] = EmbeddingStepLog(
                **asdict(embedding_log), threshold=embedding_threshold,
                model_id=embedding_model_id, model_revision=embedding_model_revision,
            )

        # Combine
        if notify_on_steps:
            print("Combining similarity filters.", flush=True)
        kept_indices = self._combine_kept_indices(tfidf_kept, embedding_kept)
        if notify_on_steps:
            print(f"Finished combining filters ({len(kept_indices)} kept).", flush=True)
        self.contamination_log.steps["combined"] = ContaminationStepLog(
            input_count=len(exact_kept),
            passed_count=len(kept_indices),
            rejected_count=len(exact_kept) - len(kept_indices),
        )

        # Sampling
        if max_samples is not None:
            available = len(kept_indices)
            if notify_on_steps:
                print(f"Starting random sampling (maximum {max_samples} texts).", flush=True)
            kept_indices = self._sample_indices(kept_indices, max_samples, random_seed)
            if notify_on_steps:
                print(f"Finished random sampling ({len(kept_indices)} selected).", flush=True)
            self.contamination_log.steps["sampling"] = ContaminationStepLog(
                input_count=available,
                passed_count=len(kept_indices),
                rejected_count=available - len(kept_indices),
            )

        self.contamination_log.selected_indices = kept_indices.copy()
        return kept_indices

    @staticmethod
    def _combine_kept_indices(tfidf_kept: list[int], embedding_kept: list[int]) -> list[int]:
        embedding_kept_set = set(embedding_kept)
        return [index for index in tfidf_kept if index in embedding_kept_set]

    @staticmethod
    def _sample_indices(kept_indices: list[int], max_samples: int, random_seed: int | None) -> list[int]:
        if max_samples >= len(kept_indices):
            return kept_indices
        return sorted(Random(random_seed).sample(kept_indices, max_samples))

    @staticmethod
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
