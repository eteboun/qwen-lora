import argparse
import json
from html import escape
from os.path import relpath
from pathlib import Path
from urllib.parse import quote


MBPP_RUNS_DIR = Path(__file__).resolve().parent / "mbpp_runs"
COMPLETION_FORMAT_LABELS = {
    "raw_python": "Raw Python",
    "single_markdown_python": "Single Python code block",
    "code_with_explanation": "Code with explanation",
    "other": "Other",
}


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
    format_suffix = ""
    if "completion_format" in example:
        format_label = COMPLETION_FORMAT_LABELS[example["completion_format"]]
        format_suffix = f" · {escape(format_label)}"
    sections = []
    if "system_prompt" in example:
        sections.append(details("System prompt", example["system_prompt"]))
    sections.extend([
        details("User prompt", example["prompt"]),
        details("Model response", example["output_text"]),
        details("Tested solution", example["solution"]),
    ])
    if "reference_solution" in example:
        sections.append(details("Reference solution", example["reference_solution"]))
    sections.extend([
        details("Input token IDs", json.dumps(input_ids)),
        details("Output token IDs", json.dumps(output_ids)),
    ])
    return f"""
    <article>
      <h3>MBPP {escape(str(example["task_id"]))}
        <span class="{'passed' if passed else 'failed'}">{'PASS' if passed else 'FAIL'}</span>
      </h3>
      <p>{len(input_ids)} input tokens · {len(output_ids)} output tokens{format_suffix}</p>
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
    format_overview = ""
    if summary_path.exists():
        summary = read_json(summary_path)
        score = f'{summary["pass_at_1"]:.1%} pass@1 · {summary["passed"]}/{summary["total"]} passed'
        if "completion_formats" in summary:
            counts = [
                (COMPLETION_FORMAT_LABELS[name], count)
                for name, count in summary["completion_formats"].items()
            ]
            counts.append(("Starts with a Markdown fence", summary["starts_with_markdown_count"]))
            format_rows = "".join(
                f'<tr><th scope="row">{escape(label)}</th><td>{count:,}</td>'
                f'<td>{count / summary["total"]:.1%}</td></tr>'
                for label, count in counts
            )
            format_overview = (
                '<h2>Response formats</h2><table><thead><tr><th scope="col">Measure</th>'
                '<th scope="col">Count</th><th scope="col">Share</th></tr></thead>'
                f'<tbody>{format_rows}</tbody></table>'
            )
    else:
        passed = sum(example["passed"] for example in examples)
        score = f'Incomplete run · {passed} passed · {len(examples)}/{config["num_examples"]} evaluated'

    dataset = config["dataset"]
    title = "MBPP trained model evaluation" if config.get("run_kind") == "sft" else "MBPP baseline"
    note = config.get("note", "").strip()
    note_html = f'<p class="run-note">{escape(note)}</p>' if note else ""
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · {escape(run_dir.name)}</title>
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
  <h1>{title}</h1>
  {note_html}
  <p><strong>{escape(score)}</strong></p>
  <p>{escape(config["model_id"])} · {escape(dataset["id"])} · {escape(dataset["config"])} / {escape(dataset["split"])}</p>
  <p>Run: <time>{escape(run_dir.name)}</time></p>
</header>
<main>
  {format_overview}
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
    html_name: str = "report.html",
) -> Path:
    log = read_json(run_dir / "contamination.json")
    datasets = {
        role: log.get(f"{role}_dataset_metadata") or {"revision": log.get(f"{role}_dataset_revision")}
        for role in ("training", "reference")
    }
    pipeline_path = run_dir / "pipeline.json"
    if pipeline_path.is_file():
        datasets = read_json(pipeline_path)["datasets"]
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
    elapsed = log.get("elapsed_seconds")
    reference_count = log.get("reference_count")
    summary_parts = []
    if reference_count is not None:
        summary_parts.append(f"{reference_count:,} reference texts")
    if elapsed is not None:
        summary_parts.append(f"{elapsed:.4f} s total")
    timing_summary = f"<p>{' · '.join(summary_parts)}</p>" if summary_parts else ""
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

    output_path = run_dir / html_name
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
  {timing_summary}
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


def write_analysis_report(analysis_path: Path) -> Path:
    """Render an aggregate HTML report from a saved analysis JSON."""
    analysis = read_json(analysis_path)
    run_dir = analysis_path.parent
    pipeline_id = analysis.get("pipeline_id")
    source = analysis.get("source")
    tokenizer_id = analysis.get("tokenizer_id")
    contamination_line = ""
    if pipeline_id is not None and (run_dir / "contamination.html").is_file():
        contamination_line = '<p><a href="contamination.html">Contamination report</a></p>'
    elif run_id := analysis.get("source_contamination_run_id"):
        log_path = analysis.get("source_contamination_log")
        contamination_html = Path(log_path).with_name("report.html") if log_path else None
        if contamination_html is not None and (run_dir / contamination_html).is_file():
            href = quote(contamination_html.as_posix())
            contamination_line = f'<p><a href="{href}">Contamination report {escape(run_id)}</a></p>'
        else:
            contamination_line = f"<p>Contamination run <code>{escape(run_id)}</code></p>"

    token_rows = []
    for name, summary in analysis["token_lengths"].items():
        cells = (
            [f"{summary['median']:,.2f}", f"{summary['p95']:,.2f}", f"{summary['p99']:,.2f}", f"{summary['maximum']:,}"]
            if summary is not None else ["—"] * 4
        )
        token_rows.append(
            f'<tr><th scope="row">{escape(name.title())}</th>'
            + "".join(f"<td>{value}</td>" for value in cells) + "</tr>"
        )

    counts = [(COMPLETION_FORMAT_LABELS[name], count) for name, count in analysis["completion_formats"].items()]
    counts.extend([
        ("Completion starts with a Markdown fence", analysis["starts_with_markdown_count"]),
        ("Prompt requests an explanation", analysis["prompt_explanation_request_count"]),
        ("Prompt requests Markdown", analysis["prompt_markdown_request_count"]),
    ])
    count_rows = "".join(
        f'<tr><th scope="row">{escape(label)}</th><td>{count:,}</td></tr>' for label, count in counts
    )
    source_line = f"<p>Source <code>{escape(str(source))}</code></p>" if source is not None else ""
    tokenizer_line = f"<p>Tokenizer <code>{escape(tokenizer_id)}</code></p>" if tokenizer_id else ""

    output_path = analysis_path.with_suffix(".html") if pipeline_id is not None else run_dir / "report.html"
    output_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Python dataset analysis</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 52rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    code {{ overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid GrayText; padding: .45rem; text-align: right; }}
    th:first-child {{ text-align: left; }}
    .table-wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Python dataset analysis</h1>
  <p><strong>{len(analysis["rows"]):,} examples</strong></p>
  {source_line}
  {tokenizer_line}
  {contamination_line}
  <h2>Token lengths</h2>
  <div class="table-wrap"><table>
    <thead><tr><th scope="col">Text</th><th scope="col">Median</th><th scope="col">P95</th>
      <th scope="col">P99</th><th scope="col">Maximum</th></tr></thead>
    <tbody>{''.join(token_rows)}</tbody>
  </table></div>
  <h2>Completion formats and prompt requests</h2>
  <table>
    <thead><tr><th scope="col">Measure</th><th scope="col">Count</th></tr></thead>
    <tbody>{count_rows}</tbody>
  </table>
</main>
</body>
</html>
""", encoding="utf-8")
    return output_path


def write_filtered_analysis_report(run_dir: Path, *, html_name: str = "report.html") -> Path:
    """Render a limited preview from a saved filtered analysis."""
    result = read_json(run_dir / "filtered.json")
    rows = result["rows"][:result["html_upper_bound"]]
    source = Path(result["source_analysis"])
    summary_path = source.with_suffix(".html") if result.get("pipeline_id") else source.with_name("report.html")
    analysis_link = ""
    if (run_dir / summary_path).is_file():
        analysis_link = (
            f'<p><a href="{quote(summary_path.as_posix())}">Analysis report {escape(source.parent.name)}</a></p>'
        )
    criteria = " · ".join(
        f"{escape(name.replace('_', ' ').capitalize())} <strong>{escape(str(value))}</strong>"
        for name, value in result["criteria"].items() if value is not None
    ) or "All rows"

    fields = (
        ("source_index", "Source index"), ("prompt_tokens", "Prompt tokens"),
        ("completion_tokens", "Completion tokens"), ("total_tokens", "Total tokens"),
        ("completion_format", "Completion format"), ("starts_with_markdown", "Starts with Markdown"),
        ("prompt_requests_explanation", "Requests explanation"),
        ("prompt_requests_markdown", "Requests Markdown"),
    )
    headings = "".join(f'<th scope="col">{label}</th>' for _, label in fields)
    table_rows = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = row.get("source_index", row.get("index")) if key == "source_index" else row[key]
            text = ("Yes" if value else "No") if isinstance(value, bool) else str(value)
            evidence_key = {
                "prompt_requests_explanation": "explanation_request_match",
                "prompt_requests_markdown": "markdown_request_match",
            }.get(key)
            evidence = row.get(evidence_key) if evidence_key else None
            if evidence:
                text += f" · {evidence}"
            cells.append(f"<td>{escape(text)}</td>")
        table_rows.append(f"<tr>{''.join(cells)}</tr>")
    output = run_dir / html_name
    output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filtered Python dataset analysis</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 74rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: .45rem; border-bottom: 1px solid GrayText; text-align: left; vertical-align: top; }}
  </style>
</head>
<body>
<main>
  <h1>Filtered Python dataset analysis</h1>
  {analysis_link}
  <p>{criteria}</p>
  <p><strong>{result["matched_count"]:,} of {result["input_count"]:,} rows matched.</strong>
    Showing {len(rows):,} in source order.</p>
  <div class="table-wrap"><table>
    <thead><tr>{headings}</tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table></div>
</main>
</body>
</html>
""", encoding="utf-8")
    return output


def write_selection_report(run_dir: Path) -> Path:
    """Render the sampling settings, counts, and source filter link."""
    log = read_json(run_dir / "selection.json")
    filter_path = Path(log["source_filter"])
    source = filter_path.with_suffix(".html")
    if not (run_dir / source).is_file():
        source = filter_path.with_name("report.html")
    source_link = (
        f'<p><a href="{quote(source.as_posix())}">Source filter report</a></p>'
        if (run_dir / source).is_file() else ""
    )
    output = run_dir / "selection.html"
    output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT sample selection</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 48rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    dt {{ font-weight: 600; }}
    dd {{ margin: 0 0 .6rem; }}
  </style>
</head>
<body>
<main>
  <h1>SFT sample selection</h1>
  <p><strong>{log["selected_count"]:,} of {log["input_count"]:,} eligible rows selected.</strong></p>
  <dl>
    <dt>Maximum samples</dt><dd>{log["max_samples"]:,}</dd>
    <dt>Random seed</dt><dd>{log["seed"]}</dd>
    <dt>Dataset</dt><dd><code>{escape(log["dataset"])}</code></dd>
  </dl>
  {source_link}
</main>
</body>
</html>
""", encoding="utf-8")
    return output


def write_memory_profile_report(profile_path: Path) -> Path:
    """Render a training memory profile from its saved JSON."""
    profile = read_json(profile_path)
    measurements = profile["measurements"]
    stages = (
        ("Before model load", "initial_memory"),
        ("Base model loaded", "base_model_initialized_memory"),
        ("Trainer initialized", "trainer_initialized_memory"),
        ("After first forward", "after_forward_memory"),
        ("After first backward", "after_backward_memory"),
        ("After first optimizer step", "after_optimizer_step_memory"),
    )
    rows = []
    for label, key in stages:
        memory = measurements[key]
        cells = (
            "".join(f"<td>{memory[metric]:.3f}</td>" for metric in ("allocated", "reserved", "peak"))
            if memory is not None else '<td colspan="3">Unavailable</td>'
        )
        rows.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')

    summary = (
        ("GPU", profile["gpu_name"]),
        ("Model", profile["model_id"]),
        ("Model revision", profile["revision"]),
        ("Model dtype", profile["dtype"]),
        ("Source dataset", profile["dataset_path"]),
        ("Source row indices (zero-based)", ", ".join(map(str, profile["source_indices"]))),
        ("First forward tokens", measurements["first_forward_token_count"]),
        ("Base model trainable parameters", measurements["base_model_trainable_params"]),
        ("LoRA trainable parameters", measurements["trainer_initialized_trainable_params"]),
    )
    entries = []
    for label, value in summary:
        display = "Unavailable" if value is None else format(value, "," if isinstance(value, int) else "")
        entries.append(f"<dt>{label}</dt><dd>{escape(display)}</dd>")
    output = profile_path.with_suffix(".html")
    if output.resolve() == profile_path.resolve():
        raise ValueError("The HTML report must have a separate output path.")
    output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training memory profile</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 58rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    dt {{ font-weight: 600; }}
    dd {{ margin: 0 0 .6rem; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    caption {{ text-align: left; font-weight: 600; }}
    th, td {{ padding: .5rem; border-bottom: 1px solid GrayText; text-align: right; }}
    th:first-child {{ text-align: left; }}
    details {{ margin-top: 1rem; }}
    pre {{ overflow: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Training memory profile</h1>
  <dl>{"".join(entries)}</dl>
  <table>
    <caption>CUDA memory (GiB)</caption>
    <thead><tr><th scope="col">Stage</th><th scope="col">Allocated</th><th scope="col">Reserved</th><th scope="col">Peak allocated</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <p>Peak tracking restarts before the training step.</p>
  {details("SFT config", json.dumps(profile["sft_config"], indent=2))}
  {details("LoRA config", json.dumps(profile["lora_config"], indent=2))}
</main>
</body>
</html>
""", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from MBPP run JSON files.")
    parser.add_argument("run_dir", nargs="?", type=Path, help="Run directory. Defaults to all MBPP runs.")
    args = parser.parse_args()
    run_dirs = [args.run_dir] if args.run_dir else [
        config_path.parent for config_path in sorted(MBPP_RUNS_DIR.glob("*/config.json"))
    ]
    if not run_dirs:
        parser.error(f"No MBPP runs found in {MBPP_RUNS_DIR}.")
    for run_dir in run_dirs:
        try:
            output = write_report(run_dir)
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"Could not generate report for {run_dir}: {error}\n")
        print(output)


if __name__ == "__main__":
    main()
