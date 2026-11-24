from __future__ import annotations

import unicodedata
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING

from score_cache import ScoreRecord, hash_texts, load_score_record, scoring_key, write_scores

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike
    from sklearn.feature_extraction.text import TfidfVectorizer


SIMILARITY_BATCH_SIZE = 256


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


class ContaminationScorer:
    def get_or_create_scores(
        self,
        training_texts: list[str],
        reference: list[str],
        vectorizer: TfidfVectorizer,
        *,
        embedding_encoder: Callable[[list[str]], ArrayLike] | None = None,
        embedding_model_id: str | None = None,
        embedding_model_revision: str | None = None,
        embedding_encode_settings: dict[str, object] | None = None,
        training_dataset_metadata: dict[str, object] | None = None,
        reference_dataset_metadata: dict[str, object] | None = None,
        notify_on_steps: bool = False,
    ) -> ScoreRecord:
        """Return cached scores or compute them from the supplied prompts."""
        dataset_metadata = {
            "training": deepcopy(training_dataset_metadata),
            "reference": deepcopy(reference_dataset_metadata),
        }
        embedding = None
        cacheable = embedding_encoder is None
        if embedding_encoder is not None:
            embedding = {
                "model_id": embedding_model_id,
                "model_revision": embedding_model_revision,
                "encode_settings": embedding_encode_settings,
            }
            cacheable = bool(
                embedding_model_id and embedding_model_revision
                and len(embedding_model_revision) == 40
                and all(char in "0123456789abcdef" for char in embedding_model_revision)
                and embedding_encode_settings is not None
            )
        key = scoring_key(training_texts, reference, vectorizer, embedding) if cacheable else None
        if key is not None:
            scoring_hash, config = key
            cached = load_score_record(scoring_hash)
            if notify_on_steps:
                print(f"Score cache {'hit' if cached is not None else 'miss'} ({scoring_hash}).", flush=True)
            if cached is not None:
                return cached
        elif notify_on_steps:
            print("Score cache skipped because the scoring configuration has no stable cache key.", flush=True)

        scores = self._generate_scores(training_texts, reference, vectorizer, embedding_encoder, notify_on_steps)
        if key is not None:
            write_scores(scoring_hash, config, scores, dataset_metadata)

        return ScoreRecord(
            scores=scores,
            training_text_hash=config["training_text_hash"] if key else hash_texts(training_texts),
            reference_text_hash=config["reference_text_hash"] if key else hash_texts(reference),
            scoring_hash=key[0] if key else None,
            cache_hit=False if key else None,
            embedding=embedding or {},
            dataset_metadata=dataset_metadata,
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
            "reference_count": len(reference),
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
