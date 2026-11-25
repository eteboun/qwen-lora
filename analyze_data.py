"""Analyze token lengths and Python completion formats in conversational SFT data."""

import ast
import json
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path
from statistics import quantiles


PYTHON_FENCE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?:python|py)?[ \t]*\r?\n"
    r"(?P<code>.*?)^[ \t]*(?P=fence)[ \t]*(?=\r?$)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
FENCE_START = re.compile(r"^(?:`{3,}|~{3,})")
PROMPT_FENCE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*(?=\r?$)",
    re.MULTILINE | re.DOTALL,
)
EXPLANATION_REQUEST = re.compile(
    r"\b(?:explain\b|describe\s+how\s+(?:your|the)\s+(?:code|solution|implementation)\b"
    r"|(?:provide|include|add|give|write|along\s+with|followed\s+by)\s+"
    r"(?:(?:a|an|the|brief|short|detailed|written|step-by-step)\s+)*"
    r"(?:explanation|rationale|reasoning|walkthrough)\b)",
    re.IGNORECASE,
)
MARKDOWN_REQUEST = re.compile(
    r"\b(?:use\s+|(?:wrap|put|enclose|format|return|output|present|provide|write|render)\s+"
    r"(?:[^.!?;\n]{0,100}?\s+)?(?:in|as|inside|within|with)\s+)"
    r"(?:(?:an?|the|your|fenced|python|markdown)\s+)*"
    r"(?:markdown\b|(?:code\s+)?fences?\b|code\s+blocks?\b|triple\s+backticks?\b)",
    re.IGNORECASE,
)
NEGATED_REQUEST = re.compile(
    r"\b(?:do\s+not|don['’]t|never|avoid|without|no\s+need\s+to|you\s+(?:need\s+not|may\s+omit))\b",
    re.IGNORECASE,
)
CODE_COMMENT_REQUEST = re.compile(
    r"\b(?:in|as|using|with|through|by\s+adding)\s+"
    r"(?:(?:the|your|inline|code|python|source|brief|clear)\s+)*(?:comments?|docstrings?)\b",
    re.IGNORECASE,
)
EXPLANATION_COMMAND_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?"
    r"(?:(?:and|then|also|please|briefly|can\s+you|could\s+you|you\s+(?:must|should)|"
    r"your\s+task\s+is\s+to)\s+)*$",
    re.IGNORECASE,
)
COMPLETION_FORMATS = ("raw_python", "single_markdown_python", "code_with_explanation", "other")


@dataclass
class RowAnalysis:
    source_index: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_format: str
    starts_with_markdown: bool
    prompt_requests_explanation: bool
    prompt_requests_markdown: bool
    explanation_request_match: str | None
    markdown_request_match: str | None


@dataclass
class TokenSummary:
    median: float
    p95: float
    p99: float
    maximum: int


@dataclass
class DataAnalysis:
    rows: list[RowAnalysis]
    token_lengths: dict[str, TokenSummary | None]
    completion_formats: dict[str, int]
    starts_with_markdown_count: int
    prompt_explanation_request_count: int
    prompt_markdown_request_count: int


def write_analysis_json(
    analysis: DataAnalysis,
    path: Path,
    *,
    source: str | Path | None = None,
    tokenizer_id: str | None = None,
    pipeline_id: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(analysis)
    payload["source"] = str(source) if source is not None else None
    payload["tokenizer_id"] = tokenizer_id
    if pipeline_id is not None:
        payload["pipeline_id"] = pipeline_id
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _is_python(text: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
            compile(tree, "<completion>", "exec")
    except (SyntaxError, ValueError):
        return False
    return any(
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, (ast.Name, ast.Constant))
        for statement in tree.body
    )


def _completion_format(text: str) -> str:
    fenced_text = text.strip()
    blocks = list(PYTHON_FENCE.finditer(fenced_text))
    if not blocks:
        return "raw_python" if _is_python(text) else "other"
    if not all(_is_python(block["code"]) for block in blocks):
        return "other"
    if PYTHON_FENCE.sub("", fenced_text).strip():
        return "code_with_explanation"
    return "single_markdown_python" if len(blocks) == 1 else "other"


def _summarize(lengths: list[int]) -> TokenSummary | None:
    if not lengths:
        return None
    if len(lengths) == 1:
        length = lengths[0]
        return TokenSummary(length, length, length, length)
    percentiles = quantiles(lengths, n=100, method="inclusive")
    return TokenSummary(percentiles[49], percentiles[94], percentiles[98], max(lengths))


def _prompt_requests(text: str) -> tuple[str | None, str | None]:
    explanation = None
    markdown = None
    prose = PROMPT_FENCE.sub("\n", text)
    for clause in re.split(r"[.!?;\n]+", prose):
        if not explanation:
            for match in EXPLANATION_REQUEST.finditer(clause):
                prefix = clause[:match.start()]
                command_prefix = re.split(r"\b(?:and|then)\b|[:,]", prefix, flags=re.IGNORECASE)[-1]
                if (
                    (EXPLANATION_COMMAND_PREFIX.fullmatch(command_prefix)
                     or re.match(r"along\s+with|followed\s+by", match[0], re.IGNORECASE))
                    and not NEGATED_REQUEST.search(prefix)
                    and not CODE_COMMENT_REQUEST.search(clause[match.end():])
                ):
                    explanation = clause[match.start():].strip()
                    break
        if not markdown:
            for match in MARKDOWN_REQUEST.finditer(clause):
                if not NEGATED_REQUEST.search(clause[:match.start()]):
                    markdown = match[0]
                    break
        if explanation and markdown:
            break
    return explanation, markdown


def analyze_data(dataset, tokenizer) -> DataAnalysis:
    """Analyze Python completions without truncation, preserving each row's source_index."""
    return summarize_analysis(analyze_rows(dataset, tokenizer))


def analyze_rows(dataset, tokenizer) -> list[RowAnalysis]:
    """Measure individual Python completion pairs without computing summary statistics."""
    rows = []
    for batch in batched(dataset, 128):
        prompts = ["\n".join(message["content"] for message in row["prompt"]) for row in batch]
        completions = ["\n".join(message["content"] for message in row["completion"]) for row in batch]
        conversations = [row["prompt"] + row["completion"] for row in batch]
        prompt_ids = tokenizer(prompts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        completion_ids = tokenizer(completions, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        chat_ids = tokenizer.apply_chat_template(
            conversations, tokenize=True, return_dict=True, truncation=False,
        )["input_ids"]
        for row, prompt, completion, chat, prompt_text, text in zip(
            batch, prompt_ids, completion_ids, chat_ids, prompts, completions, strict=True,
        ):
            explanation, markdown = _prompt_requests(prompt_text)
            rows.append(RowAnalysis(
                source_index=row["source_index"],
                prompt_tokens=len(prompt),
                completion_tokens=len(completion),
                total_tokens=len(chat),
                completion_format=_completion_format(text),
                starts_with_markdown=bool(FENCE_START.match(text.lstrip())),
                prompt_requests_explanation=explanation is not None,
                prompt_requests_markdown=markdown is not None,
                explanation_request_match=explanation,
                markdown_request_match=markdown,
            ))

    return rows


def summarize_analysis(rows: list[RowAnalysis]) -> DataAnalysis:
    """Build an analysis summary from already measured rows."""
    formats = Counter(row.completion_format for row in rows)
    return DataAnalysis(
        rows=rows,
        token_lengths={
            "prompt": _summarize([row.prompt_tokens for row in rows]),
            "completion": _summarize([row.completion_tokens for row in rows]),
            "total": _summarize([row.total_tokens for row in rows]),
        },
        completion_formats={name: formats[name] for name in COMPLETION_FORMATS},
        starts_with_markdown_count=sum(row.starts_with_markdown for row in rows),
        prompt_explanation_request_count=sum(row.prompt_requests_explanation for row in rows),
        prompt_markdown_request_count=sum(row.prompt_requests_markdown for row in rows),
    )
