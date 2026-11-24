import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import unidata_version


SCORE_CACHE_DIR = Path(__file__).resolve().parent / "score_cache"
SCORING_VERSION = 1


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _text_hash(texts):
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
            "training_text_hash": _text_hash(training_texts),
            "reference_text_hash": _text_hash(reference),
            "tfidf": params,
            "embedding": embedding_config,
        }
        serialized = _canonical_json(config)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(serialized.encode("ascii")).hexdigest(), json.loads(serialized)


def read_scores(scoring_hash):
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
    return json.loads((entry / "scores.json").read_text(encoding="utf-8"))


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
