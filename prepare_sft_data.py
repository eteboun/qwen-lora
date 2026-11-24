import json
from pathlib import Path
from tempfile import TemporaryDirectory

from contamination_filter import ContaminationFilter
from dataset_loader import load_magicoder_python_dataset
from score_cache import load_score_record


SCORING_HASH = "c80ec7e5c359ef80647f4fec3ce43c1cfad9aaa7a379f20d1253d5237bd05a54"
SAMPLE_COUNT = 5_000
RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "prepared_data" / "magicoder_python_sft.jsonl"


def prepare_sft_data(output_path: Path) -> None:
    record = load_score_record(SCORING_HASH)
    if record is None:
        raise FileNotFoundError(f"Missing contamination scores for {SCORING_HASH}.")
    dataset = load_magicoder_python_dataset()
    filterer = ContaminationFilter()
    selected = filterer.filter_scores(
        record, list(dataset["problem"]), similarity_threshold=0.45,
        embedding_threshold=0.75, max_samples=SAMPLE_COUNT, random_seed=RANDOM_SEED,
    )
    eligible = filterer.contamination_log.steps["combined"].passed_count
    if len(selected) < SAMPLE_COUNT:
        raise ValueError(f"Expected at least {SAMPLE_COUNT:,} eligible rows, got {eligible:,}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        with staged.open("w", encoding="utf-8") as output:
            for example in dataset.select(selected):
                row = {
                    "prompt": [{"role": "user", "content": example["problem"]}],
                    "completion": [{"role": "assistant", "content": example["solution"]}],
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.replace(output_path)

    print(f"Selected {len(selected):,} of {eligible:,} eligible rows. Wrote {output_path}.")


if __name__ == "__main__":
    prepare_sft_data(OUTPUT_PATH)
