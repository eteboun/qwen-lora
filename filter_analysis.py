import json
from dataclasses import asdict, dataclass
from os.path import relpath
from pathlib import Path

from analyze_data import COMPLETION_FORMATS, DataAnalysis, RowAnalysis


@dataclass
class FilteredAnalysis:
    criteria: dict[str, int | bool | str | None]
    input_count: int
    rows: list[RowAnalysis]

    @property
    def matched_count(self) -> int:
        return len(self.rows)


def filter_analysis(
    analysis: DataAnalysis,
    *,
    token_budget: int | None = None,
    starts_with_markdown: bool | None = None,
    prompt_requests_explanation: bool | None = None,
    prompt_requests_markdown: bool | None = None,
    completion_format: str | None = None,
) -> FilteredAnalysis:
    """Apply optional criteria to an analysis without retokenizing."""
    if token_budget is not None and token_budget < 1:
        raise ValueError("token_budget must be a positive token count.")
    if completion_format is not None and completion_format not in COMPLETION_FORMATS:
        raise ValueError(f"Unknown completion format: {completion_format}.")

    criteria = {
        "starts_with_markdown": starts_with_markdown,
        "prompt_requests_explanation": prompt_requests_explanation,
        "prompt_requests_markdown": prompt_requests_markdown,
        "completion_format": completion_format,
    }
    active = {key: value for key, value in criteria.items() if value is not None}
    rows = analysis.rows
    matched = [
        row for row in rows
        if (token_budget is None or row.total_tokens <= token_budget)
        and all(getattr(row, key) == value for key, value in active.items())
    ]
    return FilteredAnalysis(
        criteria={"token_budget": token_budget, **criteria},
        input_count=len(rows),
        rows=matched,
    )


def write_filtered_analysis(
    result: FilteredAnalysis,
    run_dir: Path,
    source: Path,
    *,
    pipeline_id: str | None = None,
    html_upper_bound: int = 100,
) -> Path:
    if html_upper_bound < 0:
        raise ValueError("html_upper_bound must be nonnegative.")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["matched_count"] = result.matched_count
    payload["source_analysis"] = relpath(source, run_dir)
    payload["html_upper_bound"] = html_upper_bound
    if pipeline_id is not None:
        payload["pipeline_id"] = pipeline_id

    output = run_dir / "filtered.json"
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output
