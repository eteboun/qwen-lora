import argparse
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from transformers import AutoTokenizer

from analyze_data import DataAnalysis, PYTHON_FENCE, analyze_data, analyze_rows, summarize_analysis, write_analysis_json
from contamination_filter import filter_scores
from dataset_loader import load_magicoder_python_dataset
from filter_analysis import filter_analysis, write_filtered_analysis
from project_config import MODEL_ID, MODEL_REVISION
from report import write_analysis_report, write_contamination_report, write_filtered_analysis_report, write_selection_report
from score_cache import load_score_record
from select_sft_data import SftSelection, select_sft_data, write_sft_selection


SCORING_HASH = "c80ec7e5c359ef80647f4fec3ce43c1cfad9aaa7a379f20d1253d5237bd05a54"
DEFAULT_TOKEN_BUDGET = 1_024
DEFAULT_MAX_SAMPLES = 5_000
DEFAULT_SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent


@contextmanager
def _pipeline_step(name: str, timings: dict[str, float], notify: bool):
    label = name.replace("_", " ")
    if notify:
        print(f"Starting {label}.", flush=True)
    started = perf_counter()
    yield
    elapsed = perf_counter() - started
    timings[name] = elapsed
    if notify:
        print(f"Finished {label} in {elapsed:.2f} s.", flush=True)


def prepare_analyzed_data(
    candidates: list[dict], raw_analysis: DataAnalysis, tokenizer,
) -> tuple[list[dict], DataAnalysis]:
    """Unwrap standalone Python fences and remeasure only the changed pairs."""
    prepared = candidates.copy()
    candidate_positions = {row["source_index"]: index for index, row in enumerate(candidates)}
    changed = []
    for row in raw_analysis.rows:
        if (
            row.completion_format != "single_markdown_python"
            or row.prompt_requests_explanation
            or row.prompt_requests_markdown
        ):
            continue
        position = candidate_positions[row.source_index]
        candidate = candidates[position]
        if len(candidate["completion"]) != 1:
            continue
        [message] = candidate["completion"]
        block = PYTHON_FENCE.fullmatch(message["content"].strip())
        if block is None:
            continue
        replacement = {**candidate, "completion": [{**message, "content": block["code"]}]}
        prepared[position] = replacement
        changed.append(replacement)

    if not changed:
        return prepared, raw_analysis
    updates = {row.source_index: row for row in analyze_rows(changed, tokenizer)}
    rows = [updates.get(row.source_index, row) for row in raw_analysis.rows]
    return prepared, summarize_analysis(rows)


def prepare_sft_data(
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int = DEFAULT_SEED,
    generate_html: bool = False,
    notify_on_steps: bool = True,
) -> SftSelection:
    if token_budget < 1:
        raise ValueError("token_budget must be a positive token count.")
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative.")
    pipeline_started = perf_counter()
    stage_timings = {}

    with _pipeline_step("load_inputs", stage_timings, notify_on_steps):
        record = load_score_record(SCORING_HASH)
        if record is None:
            raise FileNotFoundError(f"Missing contamination scores for {SCORING_HASH}.")
        training_metadata = record.dataset_metadata.get("training") or {}
        training_revision = training_metadata.get("revision")
        if not training_revision:
            raise ValueError("The score record must identify the Magicoder dataset revision.")
        dataset = load_magicoder_python_dataset(revision=training_revision)
        prompts = list(dataset["problem"])

    with _pipeline_step("contamination_filter", stage_timings, notify_on_steps):
        result = filter_scores(record, prompts, similarity_threshold=0.45, embedding_threshold=0.75)

    with _pipeline_step("build_candidates", stage_timings, notify_on_steps):
        kept = result.kept_indices
        candidates = [
            {
                "source_index": index,
                "prompt": [{"role": "user", "content": example["problem"]}],
                "completion": [{"role": "assistant", "content": example["solution"]}],
            }
            for index, example in zip(kept, dataset.select(kept), strict=True)
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        )

    with _pipeline_step("raw_analysis", stage_timings, notify_on_steps):
        raw_analysis = analyze_data(candidates, tokenizer)

    with _pipeline_step("prepare_analysis", stage_timings, notify_on_steps):
        prepared_candidates, prepared_analysis = prepare_analyzed_data(candidates, raw_analysis, tokenizer)

    with _pipeline_step("filter_analysis", stage_timings, notify_on_steps):
        filtered = filter_analysis(
            prepared_analysis,
            token_budget=token_budget,
            completion_format="raw_python",
            prompt_requests_explanation=False,
            prompt_requests_markdown=False,
        )

    with _pipeline_step("sample_dataset", stage_timings, notify_on_steps):
        selection = select_sft_data(prepared_candidates, filtered, max_samples=max_samples, seed=seed)

    pipeline_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = PROJECT_DIR / "pipeline_runs" / pipeline_id
    with _pipeline_step("write_artifacts", stage_timings, notify_on_steps):
        run_dir.mkdir(parents=True)
        contamination_path = run_dir / "contamination.json"
        result.log.write_json(contamination_path)

        analysis_paths = []
        for name, analysis in (("raw_analysis", raw_analysis), ("prepared_analysis", prepared_analysis)):
            analysis_path = write_analysis_json(
                analysis,
                run_dir / f"{name}.json",
                tokenizer_id=MODEL_ID,
                pipeline_id=pipeline_id,
            )
            analysis_paths.append(analysis_path)
        filter_path = write_filtered_analysis(
            filtered,
            run_dir,
            run_dir / "prepared_analysis.json",
            pipeline_id=pipeline_id,
        )
        dataset_path = write_sft_selection(selection, run_dir, filter_path)

    pipeline = {
        "pipeline_id": pipeline_id,
        "timings": {
            "total_elapsed_seconds": perf_counter() - pipeline_started,
            "stage_elapsed_seconds": stage_timings,
        },
        "datasets": {
            "training": {
                "name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "revision": training_revision,
                "selection": {
                    "split": "train", "filter": 'lang == "python"',
                    "text_column": "problem", "completion_column": "solution",
                },
            },
            "reference": record.dataset_metadata.get("reference"),
        },
        "tokenizer": {"name": MODEL_ID, "revision": MODEL_REVISION},
        "artifacts": {
            "contamination": contamination_path.name,
            "raw_analysis": analysis_paths[0].name,
            "prepared_analysis": analysis_paths[1].name,
            "filtered_analysis": filter_path.relative_to(run_dir).as_posix(),
            "selection": "selection.json",
            "sft_dataset": dataset_path.name,
        },
    }
    (run_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")

    if generate_html:
        write_contamination_report(run_dir, html_name="contamination.html")
        for analysis_path in analysis_paths:
            write_analysis_report(analysis_path)
        write_filtered_analysis_report(run_dir, html_name="filtered.html")
        write_selection_report(run_dir)
    if notify_on_steps:
        print(f"Pipeline artifacts: {run_dir}", flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze contamination-filtered Magicoder candidates.")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--notify-on-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--html", action="store_true", help="Also write HTML reports.")
    args = parser.parse_args()
    selection = prepare_sft_data(
        token_budget=args.token_budget, max_samples=args.max_samples, seed=args.seed,
        generate_html=args.html, notify_on_steps=args.notify_on_steps,
    )
    print(f"Selected {len(selection.selected):,} of {selection.input_count:,} eligible candidates.")
