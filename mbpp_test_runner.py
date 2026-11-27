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
