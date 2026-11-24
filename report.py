import argparse
import json
from html import escape
from os.path import relpath
from pathlib import Path
from urllib.parse import quote


RUNS_DIR = Path(__file__).resolve().parent / "baseline_runs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def details(label, text):
    return f"<details><summary>{escape(label)}</summary><pre>{escape(text)}</pre></details>"


def render_tests(example):
    error = example.get("execution_error")
    execution = f"<p>{escape(error)}</p>" if error else ""
    if "tests" not in example:
        return "<p>Individual test results were not recorded for this run.</p>"

    rows = []
    for test in example["tests"]:
        passed = test["passed"]
        result = test.get("error") or test["result"]
        cells = [test["assertion"], result, test["expected"], "PASS" if passed else "FAIL"]
        rows.append("<tr>" + "".join(
            f"<td>{escape(str(value)) if value is not None else '—'}</td>" for value in cells
        ) + "</tr>")
    if not rows:
        return execution or "<p>No assertions were recorded.</p>"
    return (
        execution
        + "<table><thead><tr><th scope='col'>Assertion</th><th scope='col'>Result</th>"
        + "<th scope='col'>Expected</th><th scope='col'>Status</th></tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_example(example):
    passed = example["passed"]
    input_ids = example["input_token_ids"]
    output_ids = example["output_token_ids"]
    sections = []
    if "system_prompt" in example:
        sections.append(details("System prompt", example["system_prompt"]))
    sections.extend([
        details("User prompt", example["prompt"]),
        details("Model response", example["output_text"]),
        details("Tested solution", example["solution"]),
        details("Input token IDs", json.dumps(input_ids)),
        details("Output token IDs", json.dumps(output_ids)),
    ])
    return f"""
    <article>
      <h3>MBPP {escape(str(example["task_id"]))}
        <span class="{'passed' if passed else 'failed'}">{'PASS' if passed else 'FAIL'}</span>
      </h3>
      <p>{len(input_ids)} input tokens · {len(output_ids)} output tokens</p>
      {render_tests(example)}
      {''.join(sections)}
    </article>
    """


def write_report(run_dir):
    config = read_json(run_dir / "config.json")
    examples_path = run_dir / "examples.jsonl"
    examples = []
    if examples_path.exists():
        with examples_path.open(encoding="utf-8") as results:
            examples = [json.loads(line) for line in results if line.strip()]

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        score = f'{summary["pass_at_1"]:.1%} pass@1 · {summary["passed"]}/{summary["total"]} passed'
    else:
        passed = sum(example["passed"] for example in examples)
        score = f'Incomplete run · {passed} passed · {len(examples)}/{config["num_examples"]} evaluated'

    dataset = config["dataset"]
    note = config.get("note", "").strip()
    note_html = f'<p class="run-note">{escape(note)}</p>' if note else ""
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MBPP baseline · {escape(run_dir.name)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 58rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    header p, article p {{ margin: .4rem 0; }}
    .run-note {{ font-size: 1.15rem; font-weight: 600; white-space: pre-wrap; overflow-wrap: anywhere; }}
    h1, h2, h3 {{ line-height: 1.2; }}
    article {{ border: 1px solid GrayText; border-radius: .6rem; margin: 1rem 0; padding: .9rem; }}
    article h3 {{ margin: 0; }}
    details {{ margin-top: .55rem; }}
    summary {{ cursor: pointer; }}
    pre {{ padding: .75rem; background: color-mix(in srgb, CanvasText 7%, Canvas);
           white-space: pre-wrap; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; margin: .8rem 0; }}
    th, td {{ border-bottom: 1px solid GrayText; padding: .4rem; text-align: left;
             vertical-align: top; overflow-wrap: anywhere; }}
    td {{ font-family: monospace; }}
    .passed, .failed {{ font-size: .8rem; margin-left: .4rem; padding: .2rem .45rem; border-radius: .3rem; }}
    .passed {{ background: #d3eedc; color: #174a2a; }}
    .failed {{ background: #f9dddd; color: #7c2020; }}
  </style>
</head>
<body>
<header>
  <h1>MBPP baseline</h1>
  {note_html}
  <p><strong>{escape(score)}</strong></p>
  <p>{escape(config["model_id"])} · {escape(dataset["id"])} · {escape(dataset["config"])} / {escape(dataset["split"])}</p>
  <p>Run: <time>{escape(run_dir.name)}</time></p>
</header>
<main>
  {details("Run configuration", json.dumps(config, indent=2, ensure_ascii=False))}
  <h2>Evaluated examples</h2>
  {''.join(render_example(example) for example in examples) or '<p>No examples recorded yet.</p>'}
</main>
</body>
</html>
"""
    output_path = run_dir / "report.html"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def render_dataset_metadata(datasets):
    sections = []
    for role, label in (("training", "Training dataset"), ("reference", "Reference dataset")):
        metadata = datasets.get(role)
        if not metadata:
            continue
        fields = [(key, metadata[key]) for key in ("name", "revision") if metadata.get(key)]
        fields.extend(metadata.get("selection", {}).items())
        values = " · ".join(
            f"{escape(key.replace('_', ' ').capitalize())} <code>{escape(str(value))}</code>"
            for key, value in fields
        )
        if values:
            sections.append(f"<p><strong>{label}</strong> · {values}</p>")
    return "".join(sections)


def write_score_report(
    output_path, matches_by_method, training_texts=None, reference=None, *, dataset_metadata=None,
):
    sections = []
    for name, label in (("tfidf", "TF-IDF"), ("embedding", "Embedding")):
        if name not in matches_by_method:
            continue
        matches = matches_by_method[name]
        top_matches = sorted(matches, key=lambda match: match["similarity"], reverse=True)[:100]
        lines = []
        for match in top_matches:
            training_index = match["training_index"]
            reference_index = match["reference_index"]
            text = f'Magicoder [{training_index}] · MBPP [{reference_index}] · similarity {match["similarity"]:.8g}'
            if training_texts is not None and reference is not None:
                text += f"\nMagicoder prompt:\n{training_texts[training_index]}\n\nMBPP prompt:\n{reference[reference_index]}"
            elif "training_prompt" in match:
                text += f'\nMagicoder prompt:\n{match["training_prompt"]}\n\nMBPP prompt:\n{match["reference_prompt"]}'
            lines.append(text)
        heading = f"{label} best matches ({len(top_matches):,} of {len(matches):,})"
        sections.append(details(heading, "\n\n".join(lines) or "No matches recorded."))
    output_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Similarity score report</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 58rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    details {{ margin: 1rem 0; }}
    summary {{ cursor: pointer; }}
    pre {{ max-height: 32rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<main>
  <h1>Similarity score report</h1>
  {render_dataset_metadata(dataset_metadata or {})}
  {''.join(sections)}
</main>
</body>
</html>
""", encoding="utf-8")
    return output_path


def write_contamination_report(
    run_dir: Path,
    training_texts: list[str] | None = None,
    reference: list[str] | None = None,
    *,
    scores: dict | None = None,
) -> Path:
    log = read_json(run_dir / "contamination.json")
    datasets = {
        role: log.get(f"{role}_dataset_metadata") or {"revision": log.get(f"{role}_dataset_revision")}
        for role in ("training", "reference")
    }
    score_datasets = datasets
    scoring_hash = log.get("scoring_hash")
    score_path = run_dir / "scores.html"
    if scoring_hash:
        from score_cache import SCORE_CACHE_DIR, read_scores

        score_path = SCORE_CACHE_DIR / scoring_hash / "report.html"
        if scores is None and not score_path.is_file() and training_texts is not None and reference is not None:
            scores = read_scores(scoring_hash)
    if not score_path.is_file():
        if scores is not None and training_texts is not None and reference is not None:
            if scoring_hash:
                cache_config = read_json(score_path.with_name("config.json"))
                score_datasets = cache_config.get("dataset_metadata")
                if score_datasets is None:
                    score_datasets = {
                        role: {"revision": value}
                        for role, value in cache_config.get("dataset_revisions", {}).items()
                    }
            write_score_report(
                score_path, scores["matches"], training_texts, reference, dataset_metadata=score_datasets,
            )
        elif not scoring_hash:
            legacy_matches = {
                name: step["best_matches"] for name, step in log["steps"].items() if "best_matches" in step
            }
            if legacy_matches:
                write_score_report(score_path, legacy_matches, dataset_metadata=score_datasets)
    score_link = ""
    if score_path.is_file():
        href = quote(relpath(score_path, run_dir))
        score_link = f'<p><a href="{href}">View top similarity matches</a></p>'
    cache_info = ""
    if "cache_hit" in log:
        cache_status = {
            True: "Reused cached scores",
            False: "Computed and cached scores",
            None: "Computed scores without caching",
        }[log["cache_hit"]]
        cache_key = f" · <code>{escape(scoring_hash)}</code>" if scoring_hash else ""
        generation_time = log.get("score_generation_elapsed_seconds") if not log["cache_hit"] else None
        duration = f" · {generation_time:.4f} s score generation" if generation_time is not None else ""
        cache_info = f"<p>{cache_status}{cache_key}{duration}</p>"
    steps = log["steps"]
    elapsed = log["elapsed_seconds"]
    rows = []
    for name, step in steps.items():
        label = {
            "exact_match": "Exact match",
            "tfidf": "TF-IDF",
            "embedding": "Embedding",
            "combined": "Combined (OR rejection)",
            "sampling": "Random sampling",
        }.get(name, name)
        stage_label = escape(label)
        if name == "embedding":
            model_id = step.get("model_id", log.get("embedding_model_id"))
            revision = step.get("model_revision", log.get("embedding_model_revision"))
            if model_id:
                stage_label += f"<br><small>Model: <code>{escape(model_id)}</code></small>"
            if revision:
                stage_label += f"<br><small>Revision: <code>{escape(revision)}</code></small>"
        if name == "sampling":
            max_samples = log.get("max_samples")
            random_seed = log.get("random_seed")
            if max_samples is not None:
                stage_label += f"<br><small>Maximum samples: <code>{escape(str(max_samples))}</code></small>"
            seed_label = "Not set" if random_seed is None else escape(str(random_seed))
            stage_label += f"<br><small>Random seed: <code>{seed_label}</code></small>"
        threshold = "—"
        if "threshold" in step:
            value = step["threshold"]
            threshold = "Calibration only" if value is None else f"{value:g}"
        rows.append(
            f'<tr><th scope="row">{stage_label}</th><td>{step["input_count"]:,}</td>'
            f'<td>{step["passed_count"]:,}</td><td>{step["rejected_count"]:,}</td>'
            f'<td>{threshold}</td></tr>'
        )

    output_path = run_dir / "report.html"
    output_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Contamination report · {escape(run_dir.name)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 58rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: .5rem; border-bottom: 1px solid GrayText; text-align: right; }}
    th:first-child {{ text-align: left; }}
    .table-wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Contamination report</h1>
  {render_dataset_metadata(datasets)}
  {cache_info}
  <p>{log["reference_count"]:,} reference texts · {elapsed:.4f} s total</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th scope="col">Stage</th><th scope="col">Input</th><th scope="col">Kept</th>
        <th scope="col">Rejected</th><th scope="col">Threshold</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {score_link}
</main>
</body>
</html>
""", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from baseline run JSON files.")
    parser.add_argument("run_dir", nargs="?", type=Path, help="Run directory. Defaults to all baseline runs.")
    args = parser.parse_args()
    run_dirs = [args.run_dir] if args.run_dir else [
        config_path.parent for config_path in sorted(RUNS_DIR.glob("*/config.json"))
    ]
    if not run_dirs:
        parser.error(f"No baseline runs found in {RUNS_DIR}.")
    for run_dir in run_dirs:
        try:
            output = write_report(run_dir)
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"Could not generate report for {run_dir}: {error}\n")
        print(output)


if __name__ == "__main__":
    main()
