import argparse
import json
from html import escape
from pathlib import Path


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


def write_contamination_report(run_dir: Path) -> Path:
    log = read_json(run_dir / "contamination.json")
    steps = log["steps"]
    elapsed = log["elapsed_seconds"]
    overhead = elapsed - sum(step["elapsed_seconds"] for step in steps.values())
    rows = []
    score_sections = []
    for name, step in steps.items():
        label = {"exact_match": "Exact match", "tfidf": "TF-IDF"}.get(name, name)
        threshold = f'{step["threshold"]:g}' if "threshold" in step else "—"
        rows.append(
            f'<tr><th scope="row">{escape(label)}</th><td>{step["input_count"]:,}</td>'
            f'<td>{step["passed_count"]:,}</td><td>{step["rejected_count"]:,}</td>'
            f'<td>{threshold}</td><td>{step["elapsed_seconds"]:.4f} s</td></tr>'
        )
        if "best_matches" in step:
            matches = step["best_matches"]
            top_matches = sorted(matches, key=lambda match: match["similarity"], reverse=True)[:100]
            match_text = "\n\n".join(
                f'Magicoder [{match["training_index"]}] · MBPP [{match["reference_index"]}] '
                f'· similarity {match["similarity"]:.8g}\n'
                f'Magicoder prompt:\n{match["training_prompt"]}\n\n'
                f'MBPP prompt:\n{match["reference_prompt"]}'
                for match in top_matches
            )
            heading = f"{label} best matches ({len(top_matches):,} of {len(matches):,})"
            score_sections.append(details(heading, match_text or "No matches recorded."))
        elif "similarities" in step:
            scores = step["similarities"]
            score_lines = "\n".join(f"{index}\t{score:.8g}" for index, score in enumerate(scores, start=1))
            score_text = f"Stage input row\tMaximum similarity\n{score_lines}" if scores else "No scores recorded."
            score_sections.append(details(f"{label} similarities ({len(scores):,})", score_text))

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
    summary {{ cursor: pointer; }}
    pre {{ max-height: 24rem; overflow: auto; tab-size: 24; }}
  </style>
</head>
<body>
<main>
  <h1>Contamination report</h1>
  <p>{log["reference_count"]:,} reference texts · {elapsed:.4f} s total · {overhead:.4f} s between stages and bookkeeping</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th scope="col">Stage</th><th scope="col">Input</th><th scope="col">Kept</th>
        <th scope="col">Rejected</th><th scope="col">Threshold</th><th scope="col">Duration</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {''.join(score_sections)}
  <p><a href="contamination.json">View raw JSON</a></p>
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
