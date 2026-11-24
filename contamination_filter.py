from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING

from score_cache import read_scores, scoring_key, write_scores

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike
    from sklearn.feature_extraction.text import TfidfVectorizer


SIMILARITY_BATCH_SIZE = 256


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


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
    reference_count: int = 0
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

    def filter_contamination(
        self,
        training_texts: list[str],
        reference: list[str],
        vectorizer: TfidfVectorizer,
        similarity_threshold: float = 0.45,
        notify_on_steps: bool = False,
        generate_report: bool = False,
        *,
        embedding_encoder: Callable[[list[str]], ArrayLike] | None = None,
        embedding_threshold: float | None = None,
        embedding_model_id: str | None = None,
        embedding_model_revision: str | None = None,
        embedding_encode_settings: dict[str, object] | None = None,
        max_samples: int | None = None,
        random_seed: int | None = None,
        training_dataset_metadata: dict[str, object] | None = None,
        reference_dataset_metadata: dict[str, object] | None = None,
    ) -> list[int]:
        """Return kept training indices, optionally sampled uniformly without replacement."""
        # Validation
        if not 0 < similarity_threshold <= 1:
            raise ValueError("The TF-IDF similarity threshold must be in (0, 1].")
        if embedding_threshold is not None:
            if embedding_encoder is None:
                raise ValueError("An embedding encoder is required when setting an embedding threshold.")
            if not -1 <= embedding_threshold <= 1:
                raise ValueError("The embedding similarity threshold must be in [-1, 1].")
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be nonnegative.")

        filtering_started = perf_counter()
        self.contamination_log = ContaminationLog(
            reference_count=len(reference), max_samples=max_samples, random_seed=random_seed,
            training_dataset_metadata=deepcopy(training_dataset_metadata),
            reference_dataset_metadata=deepcopy(reference_dataset_metadata),
        )

        # Score cache
        embedding_config = None
        cacheable = embedding_encoder is None
        if embedding_encoder is not None and embedding_model_id and embedding_model_revision:
            cacheable = (
                len(embedding_model_revision) == 40
                and all(char in "0123456789abcdef" for char in embedding_model_revision)
                and embedding_encode_settings is not None
            )
            embedding_config = {
                "model_id": embedding_model_id,
                "model_revision": embedding_model_revision,
                "encode_settings": embedding_encode_settings,
            }
        cache_key = scoring_key(training_texts, reference, vectorizer, embedding_config) if cacheable else None
        scores = None
        if cache_key is not None:
            scoring_hash, config = cache_key
            scores = read_scores(scoring_hash)
            self.contamination_log.scoring_hash = scoring_hash
            self.contamination_log.cache_hit = scores is not None
            if notify_on_steps:
                status = "hit" if scores is not None else "miss"
                print(f"Score cache {status} ({scoring_hash}).", flush=True)
        elif notify_on_steps:
            print("Score cache skipped because the scoring configuration has no stable cache key.", flush=True)

        # Score generation
        if scores is None:
            scores = self._generate_scores(
                training_texts, reference, vectorizer, embedding_encoder, notify_on_steps,
            )
            self.contamination_log.score_generation_elapsed_seconds = scores["elapsed_seconds"]
            if cache_key is not None:
                write_scores(scoring_hash, config, scores, {
                    "training": self.contamination_log.training_dataset_metadata,
                    "reference": self.contamination_log.reference_dataset_metadata,
                })

        # Exact match
        exact_rejected = set(scores["exact_rejected_indices"])
        exact_kept = [index for index in range(len(training_texts)) if index not in exact_rejected]
        self.contamination_log.steps["exact_match"] = ContaminationStepLog(
            input_count=len(training_texts),
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
        if embedding_encoder is not None:
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
        self.contamination_log.elapsed_seconds = perf_counter() - filtering_started

        # Report
        if generate_report:
            from report import write_contamination_report

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = Path(__file__).resolve().parent / "contamination_runs" / timestamp
            run_dir.mkdir(parents=True)
            self.contamination_log.write_json(run_dir / "contamination.json")
            write_contamination_report(run_dir, training_texts, reference, scores=scores)
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

    def _generate_scores(self, training_texts, reference, vectorizer, embedding_encoder, notify):
        started = perf_counter()
        if notify:
            print(f"Starting exact match ({len(training_texts)} texts).", flush=True)
        exact_kept = self._filter_exact(training_texts, reference)
        kept_set = set(exact_kept)
        rejected = [index for index in range(len(training_texts)) if index not in kept_set]
        if notify:
            print(f"Finished exact match ({len(exact_kept)} kept).", flush=True)

        remaining = [training_texts[index] for index in exact_kept]
        scorers = [("tfidf", "TF-IDF", lambda: self._score_tfidf(remaining, reference, vectorizer))]
        if embedding_encoder is not None:
            scorers.append((
                "embedding", "Embedding",
                lambda: self._score_embeddings(remaining, reference, embedding_encoder),
            ))

        matches = {}
        for name, label, scorer in scorers:
            if notify:
                print(f"Starting {label} scoring ({len(remaining)} texts).", flush=True)
            similarities, reference_indices = scorer()
            matches[name] = [
                {"training_index": index, "reference_index": ref, "similarity": score}
                for index, (ref, score) in zip(exact_kept, zip(reference_indices, similarities, strict=True))
            ]
            if notify:
                print(f"Finished {label} scoring.", flush=True)
        return {
            "exact_rejected_indices": rejected,
            "matches": matches,
            "elapsed_seconds": perf_counter() - started,
        }

    def _filter_exact(self, training_texts: list[str], reference: list[str]) -> list[int]:
        normalized_reference = {normalize(text) for text in reference}
        return [
            index for index, text in enumerate(training_texts)
            if normalize(text) not in normalized_reference
        ]

    def _score_tfidf(
        self,
        training_texts: list[str],
        reference: list[str],
        vectorizer: TfidfVectorizer,
    ) -> tuple[list[float], list[int]]:
        training_count = len(training_texts)
        if not training_texts or not reference:
            return [], []

        from sklearn.metrics.pairwise import cosine_similarity

        vectors = vectorizer.fit_transform(training_texts + reference)
        reference_vectors = vectors[training_count:]
        scores = []
        reference_indices = []
        for start in range(0, training_count, SIMILARITY_BATCH_SIZE):
            stop = min(start + SIMILARITY_BATCH_SIZE, training_count)
            batch = vectors[start:stop]
            similarities = cosine_similarity(batch, reference_vectors, dense_output=False)
            reference_indices.extend(similarities.argmax(axis=1).A1.tolist())
            scores.extend(similarities.max(axis=1).toarray().ravel().tolist())
        return scores, reference_indices

    def _score_embeddings(
        self, training_texts: list[str], reference: list[str],
        encoder: Callable[[list[str]], ArrayLike],
    ) -> tuple[list[float], list[int]]:
        if not training_texts or not reference:
            return [], []

        from numpy import arange, asarray
        from sklearn.metrics.pairwise import cosine_similarity

        reference_vectors = asarray(encoder(reference))
        if reference_vectors.ndim != 2 or len(reference_vectors) != len(reference):
            raise ValueError("The embedding encoder must return one vector per reference text.")
        scores = []
        reference_indices = []
        for start in range(0, len(training_texts), SIMILARITY_BATCH_SIZE):
            texts = training_texts[start:start + SIMILARITY_BATCH_SIZE]
            vectors = asarray(encoder(texts))
            if vectors.ndim != 2 or len(vectors) != len(texts):
                raise ValueError("The embedding encoder must return one vector per training text.")
            similarities = cosine_similarity(vectors, reference_vectors)
            best_indices = similarities.argmax(axis=1)
            reference_indices.extend(best_indices.tolist())
            scores.extend(similarities[arange(len(texts)), best_indices].clip(-1, 1).tolist())
        return scores, reference_indices
