"""
Prompt-Evaluation Harness for Symptom Triage (MediTriage extension)
--------------------------------------------------------------------
Runs each test case in test_cases.json through three prompting strategies
(zero-shot, few-shot, persona) against a Groq-hosted LLM, scores the result
against the expected urgency label, and logs everything to a SQLite table
for querying.

Usage:
    python run_eval.py
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from prompts import PROMPT_STRATEGIES

load_dotenv()

MODEL = "openai/gpt-oss-20b"
DB_PATH = Path(__file__).parent / "eval_results.db"
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def init_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            input_text TEXT NOT NULL,
            expected_urgency TEXT NOT NULL,
            predicted_urgency TEXT,
            predicted_action TEXT,
            correct INTEGER,
            latency_ms INTEGER,
            raw_response TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def call_llm(system: str, user: str) -> tuple[str, int]:
    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    latency_ms = int((time.time() - start) * 1000)
    return resp.choices[0].message.content, latency_ms


def parse_response(raw: str):
    """Extract {urgency, action} from the model's JSON reply, tolerant of stray text."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
        return obj.get("urgency"), obj.get("action")
    except json.JSONDecodeError:
        return None, None


def run():
    test_cases = json.loads(TEST_CASES_PATH.read_text())
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total = 0
    for case in test_cases:
        for strategy_name, build_prompt in PROMPT_STRATEGIES.items():
            prompt = build_prompt(case["input"])
            try:
                raw, latency_ms = call_llm(prompt["system"], prompt["user"])
            except Exception as e:
                print(f"  [ERROR] case {case['id']} / {strategy_name}: {e}")
                continue

            predicted_urgency, predicted_action = parse_response(raw)
            correct = int(predicted_urgency == case["expected_urgency"])

            conn.execute(
                """
                INSERT INTO eval_runs
                    (case_id, strategy, input_text, expected_urgency,
                     predicted_urgency, predicted_action, correct, latency_ms, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case["id"], strategy_name, case["input"], case["expected_urgency"],
                    predicted_urgency, predicted_action, correct, latency_ms, raw,
                ),
            )
            conn.commit()
            total += 1
            mark = "correct" if correct else "WRONG"
            print(f"case {case['id']:>2} | {strategy_name:<10} | expected={case['expected_urgency']:<9} "
                  f"predicted={str(predicted_urgency):<9} [{mark}] ({latency_ms}ms)")

    conn.close()
    print(f"\nDone. {total} runs logged to {DB_PATH}")


if __name__ == "__main__":
    run()
