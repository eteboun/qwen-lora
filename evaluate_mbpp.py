import argparse
import ast
import json
import re
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from analyze_data import COMPLETION_FORMATS, FENCE_START, classify_completion_format
from dataset_loader import load_mbpp_dataset
from project_config import MBPP_REVISION, MODEL_ID, MODEL_REVISION
from report import MBPP_RUNS_DIR, write_report


DOCKER_IMAGE = "python:3.14-slim"
SANDBOX_TIMEOUT = 15
TESTS_PASSED = 42
TEST_RUNNER_PATH = Path(__file__).resolve().with_name("mbpp_test_runner.py")

SYSTEM_PROMPT = """You are a Python programming assistant.
Write correct, executable Python code that solves the given task.
Return only Python source code.
Do not include Markdown code fences, explanations, tests, example calls, or print statements."""

USER_PROMPT_TEMPLATE = """Task:
{problem_description}

Required function signature(s):
{function_signatures}

Implement the required function(s). You may include imports and helper functions if necessary.
Return the complete implementation."""

def run_in_sandbox(docker, payload=None):
    name = f"mbpp-{uuid4().hex}"
    command = [
        docker, "run", "--rm", "--interactive", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
        "--pids-limit", "16", "--log-driver", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir", "/tmp",
    ]
    if payload is None:
        command.extend([DOCKER_IMAGE, "python", "-I", "-B", "-c", "pass"])
    else:
        command.extend([
            "--mount", f"type=bind,source={TEST_RUNNER_PATH},target=/runner.py,readonly",
            DOCKER_IMAGE, "python", "-I", "-B", "/runner.py",
        ])
    try:
        return subprocess.run(
            command,
            input=payload,
            text=True,
            stdout=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
            stderr=subprocess.PIPE if payload is None else subprocess.DEVNULL,
            timeout=SANDBOX_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            subprocess.run(
                [docker, "rm", "--force", name], capture_output=True, timeout=SANDBOX_TIMEOUT,
            )
        except subprocess.TimeoutExpired as error:
            raise SystemExit(f"Docker timed out while cleaning up container {name}.") from error
        raise


def evaluate_tests(docker, solution, example):
    payload = json.dumps([example["test_setup_code"], solution, example["test_list"]])
    try:
        result = run_in_sandbox(docker, payload)
    except subprocess.TimeoutExpired:
        return {"passed": False, "tests": [], "execution_error": "Test execution timed out."}
    if result.returncode in (125, 126, 127):
        raise SystemExit(f"Docker could not run the MBPP test container (exit code {result.returncode}).")
    if result.returncode not in (1, TESTS_PASSED) or not result.stdout:
        return {"passed": False, "tests": [], "execution_error": f"No test results (exit code {result.returncode})."}
    try:
        details = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"passed": False, "tests": [], "execution_error": "The test runner returned invalid JSON."}
    return {**details, "passed": result.returncode == TESTS_PASSED}


def extract_signatures(reference_code, tests):
    functions = [
        node for node in ast.parse(reference_code).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) > 1:
        called_names = {
            node.func.id
            for test in tests
            for node in ast.walk(ast.parse(test))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        functions = [function
         for function in functions
           if function.name in called_names]

    signatures = []
    for function in functions:
        prefix = "async def" if isinstance(function, ast.AsyncFunctionDef) else "def"
        returns = f" -> {ast.unparse(function.returns)}" if function.returns else ""
        signatures.append(f"{prefix} {function.name}({ast.unparse(function.args)}){returns}:")
    return signatures


def generate_solution(model, tokenizer, example, generation_config: GenerationConfig):
    signatures = extract_signatures(example["code"], example["test_list"])
    prompt = USER_PROMPT_TEMPLATE.format(
        problem_description=example["text"],
        function_signatures="\n".join(signatures),
    )
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs,
         generation_config=generation_config)
    input_token_ids = inputs["input_ids"][0].tolist()
    output_token_ids = output[0, len(input_token_ids):].tolist()
    answer = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    return {
        "task_id": example["task_id"],
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "input_token_ids": input_token_ids,
        "output_token_ids": output_token_ids,
        "output_text": answer,
        "completion_format": classify_completion_format(answer),
        "starts_with_markdown": bool(FENCE_START.match(answer.lstrip())),
        "solution": "\n\n".join(blocks) if blocks else answer.strip(),
        "reference_solution": example["code"],
    }


def main():
    parser = argparse.ArgumentParser(description="Measure MBPP pass@1 for the base model or a trained checkpoint.")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of examples from the test split.",
    )
    parser.add_argument(
        "--note", default="", help="Free-form note shown at the top of the HTML report.",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--adapter", type=Path, help="Local PEFT LoRA adapter directory.")
    checkpoint.add_argument("--model-path", type=Path, help="Local directory containing the full model weights.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        parser.error("--adapter must contain adapter_config.json.")
    if args.model_path is not None and not (args.model_path / "config.json").is_file():
        parser.error("--model-path must contain config.json.")

    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("Docker is required to run MBPP tests.")

    try:
        docker_probe = run_in_sandbox(docker)
    except subprocess.TimeoutExpired as error:
        raise SystemExit("Docker timed out while starting the MBPP test container.") from error
    if docker_probe.returncode:
        raise SystemExit(
            f"Could not start the MBPP test container. Check Docker and run 'docker pull {DOCKER_IMAGE}'.\n"
            f"{docker_probe.stderr.strip()}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA. Run the script in a GPU-enabled WSL session.")

    test_split = load_mbpp_dataset(revision=MBPP_REVISION)["test"]
    num_examples = min(args.limit, len(test_split))
    examples = test_split.select(range(num_examples))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    model_source = MODEL_ID if args.model_path is None else str(args.model_path.resolve())
    model_revision = MODEL_REVISION if args.model_path is None else None
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=model_revision,
        local_files_only=True,
        dtype=torch.float16,
        device_map=0,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter.resolve()), local_files_only=True)
    model.eval()
    generation_config = deepcopy(model.generation_config)
    generation_config.update(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_kind = "sft" if args.adapter is not None or args.model_path is not None else "baseline"
    run_dir = MBPP_RUNS_DIR / timestamp
    run_dir.mkdir(parents=True)

    config = {
        "note": args.note,
        "run_kind": run_kind,
        "model_id": model_source,
        "model_revision": model_revision,
        "adapter_path": str(args.adapter.resolve()) if args.adapter is not None else None,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "model_dtype": str(model.dtype),
        "dataset": {
            "id": "google-research-datasets/mbpp",
            "config": "full",
            "split": "test",
            "revision": MBPP_REVISION,
        },
        "limit": args.limit,
        "num_examples": num_examples,
        "docker_image": DOCKER_IMAGE,
        "sandbox_timeout_seconds": SANDBOX_TIMEOUT,
        "generation_config": generation_config.to_dict(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saving results to {run_dir}", flush=True)

    passed_count = 0
    completion_formats = Counter(dict.fromkeys(COMPLETION_FORMATS, 0))
    starts_with_markdown_count = 0
    with (run_dir / "examples.jsonl").open("w", encoding="utf-8") as results:
        for example in examples:
            record = generate_solution(model, tokenizer, example, generation_config)
            record.update(evaluate_tests(docker, record["solution"], example))
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
            passed_count += record["passed"]
            completion_formats[record["completion_format"]] += 1
            starts_with_markdown_count += record["starts_with_markdown"]
            status = "PASS" if record["passed"] else "FAIL"
            print(f"MBPP {example['task_id']}: {status}", flush=True)

    pass_at_1 = passed_count / num_examples
    summary = {
        "passed": passed_count,
        "total": num_examples,
        "pass_at_1": pass_at_1,
        "completion_formats": dict(completion_formats),
        "starts_with_markdown_count": starts_with_markdown_count,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"pass@1: {passed_count}/{num_examples} ({pass_at_1:.1%})")

    report_path = write_report(run_dir)
    print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
