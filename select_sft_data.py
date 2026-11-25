import json
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from random import Random

from filter_analysis import FilteredAnalysis


@dataclass
class SftSelection:
    seed: int
    max_samples: int
    input_count: int
    selected: list[dict]


def select_sft_data(
    prepared_candidates: list[dict], filtered: FilteredAnalysis, *, max_samples: int = 5_000, seed: int = 42,
) -> SftSelection:
    """Sample uniformly without replacement, then keep the original row order."""
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative.")
    candidates_by_index = {row["source_index"]: row for row in prepared_candidates}
    pool = [row.source_index for row in filtered.rows]
    if len(pool) != len(set(pool)):
        raise ValueError("Filtered rows must have unique source indices.")
    if any(index not in candidates_by_index for index in pool):
        raise ValueError("A filtered source index is missing from the prepared candidates.")

    positions = sorted(Random(seed).sample(range(len(pool)), min(max_samples, len(pool))))
    selected = [candidates_by_index[pool[position]] for position in positions]
    return SftSelection(seed=seed, max_samples=max_samples, input_count=len(pool), selected=selected)


def write_sft_selection(
    selection: SftSelection, run_dir: Path, source_filter: Path,
) -> Path:
    dataset_path = run_dir / "sft.jsonl"
    with dataset_path.open("w", encoding="utf-8") as file:
        for row in selection.selected:
            example = {"prompt": row["prompt"], "completion": row["completion"]}
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    log = {
        "pipeline_id": run_dir.name,
        "source_filter": relpath(source_filter, run_dir),
        "dataset": dataset_path.name,
        "seed": selection.seed,
        "max_samples": selection.max_samples,
        "input_count": selection.input_count,
        "selected_count": len(selection.selected),
        "selected_indices": [row["source_index"] for row in selection.selected],
    }
    (run_dir / "selection.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return dataset_path
