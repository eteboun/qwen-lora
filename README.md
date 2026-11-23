# Magicoder and MBPP overlap filtering

[dataset_loader.py](dataset_loader.py) loads the Python rows from Magicoder-OSS-Instruct-75K and the full MBPP dataset. The example below compares Magicoder problem descriptions with MBPP test prompts.

## How it works

[ContaminationFilter](contamination_filter.py) applies these steps:

1. **Exact match.** Apply Unicode NFKC, case folding, and whitespace normalization. Use a set of normalized reference prompts to detect duplicates.
2. **TF-IDF and embeddings.** Score the same remaining prompts independently with each method. TF-IDF captures lexical overlap with sparse vectors. Optional embeddings capture semantic overlap with dense vectors. Both use batched cosine similarity and record the highest-scoring reference for each prompt.
3. **Thresholds and sampling.** Reject a row when either score reaches its threshold. If `max_samples` is set, sample uniformly without replacement from the remaining rows. The filter returns their original indices.

## Example

Requires `datasets`, `scikit-learn`, and `sentence-transformers` for the embedding step.

```python
from functools import partial

from numpy import float32
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from contamination_filter import ContaminationFilter
from dataset_loader import load_magicoder_python_dataset, load_mbpp_dataset

magicoder = load_magicoder_python_dataset()
mbpp = load_mbpp_dataset()
model_id = "Alibaba-NLP/gte-modernbert-base"
revision = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
model = SentenceTransformer(
    model_id, revision=revision, device="cuda", model_kwargs={"dtype": "float32"},
)

filterer = ContaminationFilter()
indices = filterer.filter_contamination(
    list(magicoder["problem"]),
    list(mbpp["test"]["text"]),
    TfidfVectorizer(ngram_range=(1, 2), dtype=float32),
    similarity_threshold=0.45,
    embedding_encoder=partial(model.encode, batch_size=16, show_progress_bar=False),
    embedding_threshold=0.75,
    embedding_model_id=model_id,
    embedding_model_revision=revision,
    embedding_encode_settings={
        "batch_size": 16, "device": "cuda", "dtype": "float32",
        "max_seq_length": model.max_seq_length,
    },
    max_samples=512,
    random_seed=42,
    generate_report=True,
)
selected = magicoder.select(indices)
```

Set `embedding_threshold=None` to inspect scores before choosing an embedding threshold.

## Reports

The latest log is available as `filterer.contamination_log`. Optional dataset revisions can be recorded with `training_dataset_revision` and `reference_dataset_revision`.

With `generate_report=True`, [report.py](report.py) writes `contamination_runs/<UTC-timestamp>/report.html` and `contamination.json`. The HTML shows thresholds, counts, total runtime, score generation time when scores are computed, and up to 100 highest-scoring prompt pairs per method. The run log also records the scoring hash, cache status, and selected indices.

## Score cache

The filter looks up `score_cache/<scoring_hash>/` directly. Each entry contains `config.json` and `scores.json`. The key includes the ordered prompts, scoring version, TF-IDF settings, and embedding configuration. Thresholds and sampling settings can change without recomputing scores.

To cache embedding scores, supply the model ID, its full commit hash, and the settings that affect encoder output. The settings describe the encoder supplied by the caller. An embedding encoder without that metadata runs without caching. Cache files can be written with `generate_report=False`. To list stored configurations, inspect `score_cache/*/config.json`.
