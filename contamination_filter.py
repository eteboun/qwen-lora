from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


@dataclass
class ContaminationStepLog:
    input_count: int
    passed_count: int
    rejected_count: int
    elapsed_seconds: float


@dataclass
class TfidfMatch:
    training_index: int
    reference_index: int
    similarity: float
    training_prompt: str
    reference_prompt: str


@dataclass
class TfidfStepLog(ContaminationStepLog):
    threshold: float
    similarities: list[float]
    best_matches: list[TfidfMatch]


@dataclass
class ContaminationLog:
    reference_count: int = 0
    elapsed_seconds: float = 0.0
    steps: dict[str, ContaminationStepLog] = field(default_factory=dict)

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
        similarity_threshold: float = 0.85,
        notify_on_steps: bool = False,
        generate_report: bool = False,
    ) -> list[int]:
        """Return the indices of training texts that pass the contamination check."""
        if not 0 < similarity_threshold <= 1:
            raise ValueError("The similarity threshold must be in (0, 1].")
        filtering_started = perf_counter()
        self.contamination_log = ContaminationLog(reference_count=len(reference))

        if notify_on_steps:
            print(f"Starting exact match ({len(training_texts)} texts).", flush=True)
        started = perf_counter()
        exact_kept = self._filter_exact(training_texts, reference)
        exact_elapsed = perf_counter() - started
        self.contamination_log.steps["exact_match"] = ContaminationStepLog(
            input_count=len(training_texts),
            passed_count=len(exact_kept),
            rejected_count=len(training_texts) - len(exact_kept),
            elapsed_seconds=exact_elapsed,
        )
        if notify_on_steps:
            print(f"Finished exact match in {exact_elapsed:.3f}s ({len(exact_kept)} kept).", flush=True)

        remaining_texts = [training_texts[index] for index in exact_kept]
        if notify_on_steps:
            print(f"Starting TF-IDF ({len(remaining_texts)} texts).", flush=True)
        started = perf_counter()
        kept, similarities, reference_indices = self._filter_tfidf(
            remaining_texts, reference, vectorizer, similarity_threshold,
        )
        tfidf_elapsed = perf_counter() - started
        best_matches = [
            TfidfMatch(
                training_index=exact_kept[index],
                reference_index=reference_index,
                similarity=similarity,
                training_prompt=remaining_texts[index],
                reference_prompt=reference[reference_index],
            )
            for index, (similarity, reference_index) in enumerate(zip(similarities, reference_indices, strict=True))
        ]
        self.contamination_log.steps["tfidf"] = TfidfStepLog(
            threshold=similarity_threshold,
            input_count=len(remaining_texts),
            passed_count=len(kept),
            rejected_count=len(remaining_texts) - len(kept),
            elapsed_seconds=tfidf_elapsed,
            similarities=similarities,
            best_matches=best_matches,
        )
        if notify_on_steps:
            print(f"Finished TF-IDF in {tfidf_elapsed:.3f}s ({len(kept)} kept).", flush=True)
        kept_indices = [exact_kept[index] for index in kept]
        self.contamination_log.elapsed_seconds = perf_counter() - filtering_started
        if generate_report:
            from report import write_contamination_report

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = Path(__file__).resolve().parent / "contamination_runs" / timestamp
            run_dir.mkdir(parents=True)
            self.contamination_log.write_json(run_dir / "contamination.json")
            write_contamination_report(run_dir)
        return kept_indices

    def _filter_exact(self, training_texts: list[str], reference: list[str]) -> list[int]:
        normalized_reference = {normalize(text) for text in reference}
        return [
            index for index, text in enumerate(training_texts)
            if normalize(text) not in normalized_reference
        ]

    def _filter_tfidf(
        self,
        training_texts: list[str],
        reference: list[str],
        vectorizer: TfidfVectorizer,
        similarity_threshold: float,
    ) -> tuple[list[int], list[float], list[int]]:
        """Return kept indices, maximum similarities, and their reference indices."""
        training_count = len(training_texts)
        if not training_texts or not reference:
            return list(range(training_count)), [], []

        from sklearn.metrics.pairwise import cosine_similarity

        vectors = vectorizer.fit_transform(training_texts + reference)
        reference_vectors = vectors[training_count:]
        batch_size = 256
        kept = []
        scores = []
        reference_indices = []
        for start in range(0, training_count, batch_size):
            stop = min(start + batch_size, training_count)
            similarities = cosine_similarity(vectors[start:stop], reference_vectors, dense_output=False)
            best_matches = similarities.max(axis=1).toarray().ravel().tolist()
            reference_indices.extend(similarities.argmax(axis=1).A1.tolist())
            scores.extend(best_matches)
            kept.extend(
                start + offset
                for offset, score in enumerate(best_matches)
                if score < similarity_threshold
                and not isclose(score, similarity_threshold, rel_tol=1e-12)
            )
        return kept, scores, reference_indices
