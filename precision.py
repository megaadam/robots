#!/usr/bin/env python3
"""Benchmark the accuracy ("precision") of an OpenAI-compatible chat model.

Self-contained: the test questions are embedded in this file (no downloads).
Covers four categories:
  - knowledge : multiple-choice (A/B/C/D)
  - math      : word problems, scored by final numeric answer
  - code      : function generation, scored by running hidden test cases
  - reasoning : logic puzzles, scored by answer match

Usage:
  ./precision.py
  ./precision.py --categories math,code
  ./precision.py --limit 5 --show
  ./precision.py --concurrency 4
  ./precision.py -v
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_BASE = "http://127.0.0.1:2001/v1"
DEFAULT_MODEL = "halogen-qwen3.8-27b"

# ---------------- question banks ----------------

KNOWLEDGE = [
    ("What is the capital of France?", ["Berlin", "Paris", "Rome", "Madrid"], "B"),
    ("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"], "B"),
    ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Pacific", "Arctic"], "C"),
    ("Who wrote 'Romeo and Juliet'?", ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"], "B"),
    ("What is the chemical symbol for gold?", ["Go", "Gd", "Au", "Ag"], "C"),
    ("Which gas do plants primarily absorb for photosynthesis?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "C"),
    ("What is the approximate speed of light in a vacuum?", ["300,000 km/s", "150,000 km/s", "500,000 km/s", "1,000,000 km/s"], "A"),
    ("What is the smallest prime number?", ["0", "1", "2", "3"], "C"),
    ("In what year did World War II end?", ["1943", "1945", "1947", "1950"], "B"),
    ("Which element has the atomic number 1?", ["Helium", "Oxygen", "Hydrogen", "Carbon"], "C"),
    ("What is the currency of Japan?", ["Won", "Yuan", "Yen", "Ringgit"], "C"),
    ("How many continents are there on Earth?", ["5", "6", "7", "8"], "C"),
]

MATH = [
    ("A train travels 60 miles in 30 minutes. How many miles does it travel in 2 hours?", 240),
    ("Apples cost $2 each. How much do 15 apples cost?", 30),
    ("What is 15% of 200?", 30),
    ("The sum of two numbers is 50. If one number is 18, what is the other?", 32),
    ("A rectangle has length 12 and width 5. What is its area?", 60),
    ("If 3 workers build 3 walls in 3 days, how many days do 12 workers take to build 12 walls?", 3),
    ("What is the average of 4, 8, 12, 16, and 20?", 12),
    ("A book costs $40 after a 20% discount. What was the original price?", 50),
    ("What is 7 x 8 - 6?", 50),
    ("If x + 5 = 12, what is x?", 7),
    ("A car uses 8 liters of fuel per 100 km. How many liters for 350 km?", 28),
    ("What is the next number in the sequence: 2, 6, 12, 20, 30, ...?", 42),
]

CODE = [
    {"name": "is_even", "prompt": "Write a Python function is_even(n) that returns True if n is even, otherwise False.",
     "tests": [([2], True), ([3], False), ([0], True)]},
    {"name": "sum_list", "prompt": "Write a Python function sum_list(lst) that returns the sum of all numbers in the list lst.",
     "tests": [([ [1, 2, 3] ], 6), ([ [] ], 0), ([ [-1, 1, 5] ], 5)]},
    {"name": "reverse_string", "prompt": "Write a Python function reverse_string(s) that returns the string s reversed.",
     "tests": [([ "abc" ], "cba"), ([ "" ], ""), ([ "ab" ], "ba")]},
    {"name": "count_vowels", "prompt": "Write a Python function count_vowels(s) that returns the number of vowels (a,e,i,o,u, case-insensitive) in s.",
     "tests": [([ "hello" ], 2), ([ "xyz" ], 0), ([ "AEIOU" ], 5)]},
    {"name": "fib", "prompt": "Write a Python function fib(n) that returns the nth Fibonacci number, where fib(0)=0 and fib(1)=1.",
     "tests": [([0], 0), ([1], 1), ([6], 8), ([10], 55)]},
    {"name": "is_palindrome", "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome, ignoring case and spaces.",
     "tests": [([ "racecar" ], True), ([ "hello" ], False), ([ "A man a plan a canal Panama" ], True)]},
    {"name": "max_of", "prompt": "Write a Python function max_of(lst) that returns the largest number in the list lst.",
     "tests": [([ [1, 5, 3] ], 5), ([ [-1, -5, -3] ], -1), ([ [7] ], 7)]},
    {"name": "factorial", "prompt": "Write a Python function factorial(n) that returns n! (the factorial of n).",
     "tests": [([0], 1), ([1], 1), ([5], 120), ([6], 720)]},
    {"name": "two_sum", "prompt": "Write a Python function two_sum(nums, target) that returns the two indices whose values add up to target. Assume exactly one solution.",
     "tests": [([ [2, 7, 11, 15], 9 ], [0, 1]), ([ [3, 2, 4], 6 ], [1, 2]), ([ [1, 5, 3], 4 ], [0, 2])]},
    {"name": "is_prime", "prompt": "Write a Python function is_prime(n) that returns True if n is a prime number, otherwise False.",
     "tests": [([2], True), ([15], False), ([97], True), ([1], False)]},
]

REASONING = [
    ("All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops definitely Lazzies? Answer yes or no.", "yes", "word"),
    ("If it rains, the ground gets wet. The ground is not wet. Did it rain? Answer yes or no.", "no", "word"),
    ("Mary has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Mary have?", "1", "number"),
    ("A farmer has 17 sheep. All but 9 run away. How many sheep are left?", "9", "number"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets?", "5", "number"),
    ("Tom is taller than Jim. Jim is taller than Sad. Who is the shortest?", "sad", "word"),
    ("I have more books than you, and you have more books than Amy. Who has the fewest books?", "amy", "word"),
    ("A clock shows exactly 3:15. What is the angle in degrees between the hour and minute hands?", "7.5", "number"),
    ("Rearrange the letters in 'IFARCILANO' to spell a US state. What state is it?", "california", "word"),
    ("What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ...?", "21", "number"),
]

# ---------------- helpers ----------------

def chat(base, model, prompt, api_key, temp, max_tokens=400):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
    }
    try:
        r = requests.post(f"{base}/chat/completions", json=payload,
                          headers=headers, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {type(e).__name__}: {e}"


def _to_num(s):
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def extract_letter(text):
    t = text.upper()
    m = re.findall(r"\b([A-D])\b", t)
    if m:
        return m[0]
    m = re.findall(r"([A-D])", t)
    return m[0] if m else None


def extract_number(text):
    m = re.search(r"answer\s*[:\-]?\s*\$?\s*([-\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        v = _to_num(m.group(1))
        if v is not None:
            return v
    nums = re.findall(r"[-\d,]+\.?\d*", text)
    if nums:
        return _to_num(nums[-1])
    return None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def format_code_tests(item):
    lines = []
    for args, exp in item["tests"]:
        arg_str = ", ".join(repr(a) for a in args)
        lines.append(f"{item['name']}({arg_str}) == {exp!r}")
    return "\n".join(lines)


def _indent(text, prefix):
    lines = str(text).splitlines()
    if not lines:
        return ""
    return lines[0] + "".join("\n" + prefix + line for line in lines[1:])


def run_code_test_detail(name, code, tests, timeout=10):
    """Run the hidden tests in a subprocess.

    Returns (rows, stderr):
      rows   - list of dicts, one per test:
               {"args": [...], "exp": <expected>, "actual": <str>, "ok": <bool>}
               "actual" is the repr of the returned value, or
               "Exception: <Type>: <msg>" if the call raised.
               Empty if the code could not be executed at all.
      stderr - the subprocess stderr (useful when rows is empty).
    """
    # Embed a Python literal (repr), not JSON: the harness is Python source,
    # so expected values must be True/False/None, not true/false/null.
    tests_literal = repr(tests)
    harness = code + "\n\n"
    harness += "import json\n"
    harness += "_TESTS = " + tests_literal + "\n"
    harness += "def _eq(a, b):\n"
    harness += "    try:\n"
    harness += "        return bool(a == b)\n"
    harness += "    except Exception:\n"
    harness += "        pass\n"
    harness += "    try:\n"
    harness += "        return sorted(map(str, a)) == sorted(map(str, b))\n"
    harness += "    except Exception:\n"
    harness += "        return False\n"
    harness += "def _check():\n"
    harness += "    for _args, _exp in _TESTS:\n"
    harness += "        try:\n"
    harness += "            _got = " + name + "(*_args)\n"
    harness += "            _actual = repr(_got)\n"
    harness += "        except Exception as _e:\n"
    harness += "            _got = None\n"
    harness += "            _actual = 'Exception: ' + type(_e).__name__ + ': ' + str(_e)\n"
    harness += "        print(json.dumps({'args': _args, 'exp': _exp, 'actual': _actual, 'ok': _eq(_got, _exp)}))\n"
    harness += "_check()\n"
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(harness)
        r = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return [], f"subprocess error: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    rows = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows, r.stderr


def run_code_test(name, code, tests, timeout=10):
    rows, _ = run_code_test_detail(name, code, tests, timeout)
    return len(rows) == len(tests) and all(row["ok"] for row in rows)


# ---------------- scorers ----------------

def score_math(ans, resp):
    got = extract_number(resp)
    if got is None:
        return False
    return abs(got - ans) < 0.01


def score_reasoning(expected, kind, resp):
    t = resp.lower().strip()
    if kind == "word":
        return expected.lower() in t
    if expected in t:
        return True
    got = extract_number(resp)
    if got is not None:
        try:
            return abs(got - float(expected)) < 0.01
        except Exception:
            return False
    return False


# ---------------- task building ----------------

def format_knowledge(q, opts):
    letters = "ABCD"
    lines = [q, ""]
    for i, o in enumerate(opts):
        lines.append(f"{letters[i]}) {o}")
    lines.append("")
    lines.append("Respond with only the single letter (A, B, C, or D) of the correct answer.")
    return "\n".join(lines)


def build_tasks(categories, limit):
    tasks = []
    lim = limit if limit and limit > 0 else None
    if "knowledge" in categories:
        for q, opts, correct in (KNOWLEDGE[:lim] if lim else KNOWLEDGE):
            prompt = format_knowledge(q, opts)
            tasks.append(("knowledge", prompt,
                          lambda resp, c=correct: extract_letter(resp) == c,
                          correct,
                          lambda resp: extract_letter(resp)))
    if "math" in categories:
        for q, ans in (MATH[:lim] if lim else MATH):
            prompt = (q + "\nShow your reasoning briefly, then give the final "
                     "numeric answer on its own line starting with 'Answer:'.")
            tasks.append(("math", prompt,
                          lambda resp, a=ans: score_math(a, resp),
                          ans,
                          lambda resp: extract_number(resp)))
    if "code" in categories:
        for item in (CODE[:lim] if lim else CODE):
            prompt = item["prompt"] + "\nRespond with only the Python code in a single ```python code block."
            state = {}

            def _scorer(resp, it=item, st=state):
                st["rows"], st["stderr"] = run_code_test_detail(
                    it["name"], extract_code(resp), it["tests"])
                rows = st["rows"]
                return len(rows) == len(it["tests"]) and all(r["ok"] for r in rows)

            def _actual(resp, it=item, st=state):
                if "rows" not in st:
                    st["rows"], st["stderr"] = run_code_test_detail(
                        it["name"], extract_code(resp), it["tests"])
                rows = st["rows"]
                if not rows:
                    err = st.get("stderr", "").strip()
                    return "(code did not run)" + (f"\n{err}" if err else "")
                lines = []
                for row in rows:
                    arg_str = ", ".join(repr(a) for a in row["args"])
                    lines.append(f"{it['name']}({arg_str}) -> {row['actual']}")
                return "\n".join(lines)

            tasks.append(("code", prompt, _scorer, format_code_tests(item), _actual))
    if "reasoning" in categories:
        for q, expected, kind in (REASONING[:lim] if lim else REASONING):
            prompt = q + "\nAnswer concisely."
            tasks.append(("reasoning", prompt,
                          lambda resp, e=expected, k=kind: score_reasoning(e, k, resp),
                          expected,
                          lambda resp: " ".join(resp.split())[:200]))
    return tasks


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--categories", default="knowledge,math,code,reasoning")
    ap.add_argument("--limit", type=int, default=0, help="max questions per category (0=all)")
    ap.add_argument("-t", "--test", type=int, default=0, help="run only the test at this 1-based index")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--show", action="store_true", help="print each question and answer")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print expected/actual values for failed questions")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    tasks = build_tasks(cats, args.limit)
    if not tasks:
        print("No questions selected.")
        raise SystemExit(1)
    if args.test:
        if not 1 <= args.test <= len(tasks):
            print(f"Test index {args.test} out of range (1-{len(tasks)}).")
            raise SystemExit(1)
        tasks = [tasks[args.test - 1]]

    print(f"endpoint : {args.base}")
    print(f"model    : {args.model}")
    print(f"questions: {len(tasks)} across {', '.join(cats)}")
    print("-" * 60)

    def run_task(task):
        cat, prompt, scorer, expected, actual_fn = task
        resp = chat(args.base, args.model, prompt, args.api_key, args.temp)
        if resp.startswith("__ERROR__"):
            return (cat, prompt, resp, False, True, expected, actual_fn)
        try:
            ok = bool(scorer(resp))
        except Exception:
            ok = False
        return (cat, prompt, resp, ok, False, expected, actual_fn)

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            results = [f.result() for f in as_completed(futs)]
    else:
        results = [run_task(t) for t in tasks]

    for i, (cat, prompt, resp, ok, is_err, expected, actual_fn) in enumerate(results, 1):
        mark = "ERR " if is_err else ("PASS" if ok else "FAIL")
        print(f"  [{i:>2}/{len(results)}] {cat:<10} {mark}")

    per = defaultdict(lambda: [0, 0])
    for cat, prompt, resp, ok, is_err, expected, actual_fn in results:
        per[cat][1] += 1
        if ok:
            per[cat][0] += 1
    total_correct = sum(v[0] for v in per.values())
    total = sum(v[1] for v in per.values())

    print("-" * 60)
    for cat in cats:
        if cat in per:
            c, t = per[cat]
            print(f"[{cat:<10}] {c}/{t}  ({100 * c / t:.1f}%)")
    print("-" * 60)
    print(f"[{'OVERALL':<10}] {total_correct}/{total}  ({100 * total_correct / total:.1f}%)")

    if args.show:
        print("\n" + "=" * 60)
        print("DETAIL")
        for cat, prompt, resp, ok, is_err, expected, actual_fn in results:
            mark = "PASS" if ok else ("ERR " if is_err else "FAIL")
            print(f"\n[{mark}] {cat}")
            print("Q:", " ".join(prompt.split())[:240])
            print("A:", " ".join(resp.split())[:300])

    if args.verbose:
        failures = [r for r in results if not r[3]]
        if failures:
            print("\n" + "=" * 60)
            print("FAILURES (expected vs actual)")
            for cat, prompt, resp, ok, is_err, expected, actual_fn in failures:
                try:
                    actual = actual_fn(resp)
                except Exception:
                    actual = None
                print(f"\n[{cat}]")
                print(f"  expected: {_indent(expected, ' ' * 12)}")
                print(f"  actual  : {_indent(actual, ' ' * 12)}")


if __name__ == "__main__":
    main()
