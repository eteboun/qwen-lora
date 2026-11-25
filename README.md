# Magicoder and MBPP overlap filtering

[dataset_loader.py](dataset_loader.py) loads the Python rows from Magicoder-OSS-Instruct-75K and the full MBPP dataset. The example below compares Magicoder problem descriptions with MBPP test prompts.

## How it works

[`get_or_create_scores()`](contamination_scorer.py) produces scores. [`filter_scores()`](contamination_filter.py) applies thresholds and returns the remaining row indices and a run log.

1. **Exact match.** Apply Unicode NFKC, case folding, and whitespace normalization. Use a set of normalized reference prompts to detect duplicates.
2. **TF-IDF and embeddings.** Score the same remaining prompts independently with each method. TF-IDF captures lexical overlap with sparse vectors. Optional embeddings capture semantic overlap with dense vectors. Both use batched cosine similarity and record the highest-scoring reference for each prompt.
3. **Thresholds.** Reject a row when either score reaches its threshold. The filter returns all remaining original indices.

## Example

Requires `datasets`, `scikit-learn`, and `sentence-transformers` for the embedding step.

```python
from functools import partial

from numpy import float32
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from contamination_filter import filter_scores
from contamination_scorer import get_or_create_scores
from dataset_loader import load_magicoder_python_dataset, load_mbpp_dataset

magicoder = load_magicoder_python_dataset()
mbpp = load_mbpp_dataset()
model_id = "Alibaba-NLP/gte-modernbert-base"
revision = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
model = SentenceTransformer(
    model_id, revision=revision, device="cuda", model_kwargs={"dtype": "float32"},
)

training_texts = list(magicoder["problem"])
reference = list(mbpp["test"]["text"])
record = get_or_create_scores(
    training_texts,
    reference,
    TfidfVectorizer(ngram_range=(1, 2), dtype=float32),
    embedding_encoder=partial(model.encode, batch_size=16, show_progress_bar=False),
    embedding_model_id=model_id,
    embedding_model_revision=revision,
    embedding_encode_settings={
        "batch_size": 16, "device": "cuda", "dtype": "float32",
        "max_seq_length": model.max_seq_length,
    },
    training_dataset_metadata={
        "name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "selection": {"split": "train", "text_column": "problem", "filter": 'lang == "python"'},
    },
    reference_dataset_metadata={
        "name": "google-research-datasets/mbpp",
        "selection": {"config": "full", "split": "test", "text_column": "text"},
    },
)
result = filter_scores(
    record,
    training_texts,
    reference=reference,
    similarity_threshold=0.45,
    embedding_threshold=0.75,
)
selected = magicoder.select(result.kept_indices)
```

Set `embedding_threshold=None` to inspect scores before choosing an embedding threshold.

## Reports

Each filter result includes its own log at `result.log`. The optional dataset metadata dictionaries can include a top-level `revision` alongside `name` and `selection`. Cache entries record the metadata supplied when scores were first written. Metadata is excluded from the scoring hash.

Filtering keeps the log in memory. To create a report, save the log as `contamination.json` in a run directory, then call [`write_contamination_report(run_dir, training_texts, reference, scores=record.scores)`](report.py). The HTML shows thresholds, counts, runtime, and a link to the top-scoring prompt pairs. The JSON records the scoring hash, cache status, and retained indices.

## Score cache

The scorer looks up `score_cache/<scoring_hash>/` directly. Each entry contains `config.json` and `scores.json`. Use `score_cache.load_score_record(hash)` to load a known entry without a model or vectorizer. Report generation adds a shared `report.html` to the entry. For an uncached run, the score report is saved as `scores.html` next to the run report. The key includes the ordered prompts, scoring version, TF-IDF settings, and embedding configuration. Thresholds and sampling settings can change without recomputing scores.

To cache embedding scores, supply the model ID, its full commit hash, and the settings that affect encoder output. The settings describe the encoder supplied by the caller. An embedding encoder without that metadata runs without caching. Score caching is independent of report generation. To list stored configurations, inspect `score_cache/*/config.json`.
