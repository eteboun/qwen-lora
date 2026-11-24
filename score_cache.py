import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import unidata_version


SCORE_CACHE_DIR = Path(__file__).resolve().parent / "score_cache"
SCORING_VERSION = 1


@dataclass
class ScoreRecord:
    scores: dict
    training_text_hash: str
    reference_text_hash: str
    scoring_hash: str | None
    cache_hit: bool | None
    embedding: dict
    dataset_metadata: dict


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def hash_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(_canonical_json(text).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def scoring_key(training_texts, reference, vectorizer, embedding_config):
    from numpy import dtype
    from sklearn.feature_extraction.text import TfidfVectorizer

    # A custom vectorizer or callable parameter has no stable JSON identity.
    if type(vectorizer) is not TfidfVectorizer:
        return None
    params = vectorizer.get_params(deep=False)
    try:
        params["dtype"] = dtype(params["dtype"]).name
        config = {
            "version": SCORING_VERSION,
            "unicode_version": unidata_version,
            "libraries": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
            "training_text_hash": hash_texts(training_texts),
            "reference_text_hash": hash_texts(reference),
            "tfidf": params,
            "embedding": embedding_config,
        }
        serialized = _canonical_json(config)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(serialized.encode("ascii")).hexdigest(), json.loads(serialized)


def _read_entry(scoring_hash):
    if len(scoring_hash) != 64 or any(char not in "0123456789abcdef" for char in scoring_hash):
        raise ValueError("Invalid scoring cache hash.")
    entry = SCORE_CACHE_DIR / scoring_hash
    if not entry.is_dir():
        return None
    config = json.loads((entry / "config.json").read_text(encoding="utf-8"))
    if config["scoring_hash"] != scoring_hash or hashlib.sha256(
        _canonical_json(config["scoring_config"]).encode("ascii")
    ).hexdigest() != scoring_hash:
        raise ValueError(f"Invalid scoring configuration in {entry}.")
    return config, json.loads((entry / "scores.json").read_text(encoding="utf-8"))


def read_scores(scoring_hash):
    entry = _read_entry(scoring_hash)
    return entry[1] if entry is not None else None


def load_score_record(scoring_hash: str) -> ScoreRecord | None:
    entry = _read_entry(scoring_hash)
    if entry is None:
        return None
    metadata, scores = entry
    config = metadata["scoring_config"]
    return ScoreRecord(
        scores=scores,
        training_text_hash=config["training_text_hash"],
        reference_text_hash=config["reference_text_hash"],
        scoring_hash=scoring_hash,
        cache_hit=True,
        embedding=config["embedding"] or {},
        dataset_metadata=metadata.get("dataset_metadata") or {
            role: {"revision": value} for role, value in metadata.get("dataset_revisions", {}).items()
        },
    )


def write_scores(scoring_hash, config, scores, dataset_metadata):
    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = SCORE_CACHE_DIR / scoring_hash
    metadata = {
        "scoring_hash": scoring_hash,
        "scoring_config": config,
        "dataset_metadata": dataset_metadata,
    }
    with TemporaryDirectory(dir=SCORE_CACHE_DIR) as temporary:
        staging = Path(temporary) / "entry"
        staging.mkdir()
        (staging / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (staging / "scores.json").write_text(_canonical_json(scores) + "\n", encoding="utf-8")
        staging.rename(destination)
