import argparse
import ast
import json
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dataset_loader import load_mbpp_dataset
from report import write_report


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DOCKER_IMAGE = "python:3.14-slim"
SANDBOX_TIMEOUT = 15
TESTS_PASSED = 42

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

# Run the solution and MBPP assertions in a fresh process for each task.
TEST_RUNNER = """
import ast
import contextlib
import json
import os
import reprlib
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
setup, solution, tests = json.load(sys.stdin)
namespace = {"__name__": "__mbpp__"}

def evaluate(expression):
    return eval(compile(ast.Expression(expression), "<mbpp>", "eval"), namespace)

def describe_error(error):
    return f"{type(error).__name__}: {reprlib.repr(str(error))}"

def compare(source):
    record = {"assertion": source, "result": None, "expected": None, "passed": False}
    try:
        tree = ast.parse(source)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
            condition = tree.body[0].test
            if isinstance(condition, ast.Compare) and len(condition.ops) == 1 and isinstance(condition.ops[0], ast.Eq):
                record["expected"] = ast.unparse(condition.comparators[0])
                actual = evaluate(condition.left)
                expected = evaluate(condition.comparators[0])
                record["passed"] = bool(actual == expected)
                record.update(result=reprlib.repr(actual), expected=reprlib.repr(expected))
            else:
                record["expected"] = "True"
                record["passed"] = bool(evaluate(condition))
                record["result"] = str(record["passed"])
        else:
            record["expected"] = "No exception"
            exec(compile(tree, "<mbpp>", "exec"), namespace)
            record.update(result="No exception", passed=True)
    except Exception as error:
        record["error"] = describe_error(error)
    return record

records = []
execution_error = None
with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    try:
        for source in (setup, solution):
            exec(compile(source, "<mbpp>", "exec"), namespace)
    except Exception as error:
        execution_error = describe_error(error)
    else:
        records = [compare(source) for source in tests]

print(json.dumps({"tests": records, "execution_error": execution_error}))
sys.exit(42 if execution_error is None and all(test["passed"] for test in records) else 1)
"""


def run_in_sandbox(docker, script, payload=None):
    name = f"mbpp-{uuid4().hex}"
    command = [
        docker, "run", "--rm", "--interactive", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
        "--pids-limit", "16", "--log-driver", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir", "/tmp", DOCKER_IMAGE, "python", "-I", "-B", "-c", script,
    ]
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
        result = run_in_sandbox(docker, TEST_RUNNER, payload)
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
        functions = [function for function in functions if function.name in called_names]

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
        output = model.generate(**inputs, generation_config=generation_config)
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
        "solution": "\n\n".join(blocks) if blocks else answer.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure MBPP pass@1 before SFT.")
    parser.add_argument("--limit", type=int, default=10, help="Number of examples from the test split.")
    parser.add_argument("--note", default="", help="Free-form note shown at the top of the HTML report.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("Docker is required to run MBPP tests.")
    try:
        probe = run_in_sandbox(docker, "pass")
    except subprocess.TimeoutExpired as error:
        raise SystemExit("Docker timed out while starting the MBPP test container.") from error
    if probe.returncode:
        raise SystemExit(
            f"Could not start the MBPP test container. Check Docker and run 'docker pull {DOCKER_IMAGE}'.\n"
            f"{probe.stderr.strip()}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA. Run the script in a GPU-enabled WSL session.")

    test_split = load_mbpp_dataset()["test"]
    examples = test_split.select(range(min(args.limit, len(test_split))))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=torch.float16, device_map=0,
    ).eval()
    generation_config = deepcopy(model.generation_config)
    generation_config.update(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(__file__).resolve().parent / "baseline_runs" / timestamp
    run_dir.mkdir(parents=True)
    config = {
        "note": args.note,
        "model_id": MODEL_ID,
        "model_dtype": str(model.dtype),
        "dataset": {"id": "google-research-datasets/mbpp", "config": "full", "split": "test"},
        "limit": args.limit,
        "num_examples": len(examples),
        "docker_image": DOCKER_IMAGE,
        "sandbox_timeout_seconds": SANDBOX_TIMEOUT,
        "generation_config": generation_config.to_dict(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saving results to {run_dir}", flush=True)

    passed = 0
    with (run_dir / "examples.jsonl").open("w", encoding="utf-8") as results:
        for example in examples:
            record = generate_solution(model, tokenizer, example, generation_config)
            record.update(evaluate_tests(docker, record["solution"], example))
            success = record["passed"]
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
            passed += success
            print(f"MBPP {example['task_id']}: {'PASS' if success else 'FAIL'}", flush=True)

    pass_at_1 = passed / len(examples)
    summary = {"passed": passed, "total": len(examples), "pass_at_1": pass_at_1}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"pass@1: {passed}/{len(examples)} ({pass_at_1:.1%})")
    print(f"HTML report: {write_report(run_dir)}")


if __name__ == "__main__":
    main()
