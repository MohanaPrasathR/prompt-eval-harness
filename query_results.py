"""
Query the eval_results.db SQLite table produced by run_eval.py.
Answers: which prompting strategy performs best, and where does each fail?
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "eval_results.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=== Accuracy by prompting strategy ===")
    rows = conn.execute(
        """
        SELECT strategy,
               COUNT(*) AS total,
               SUM(correct) AS correct,
               ROUND(100.0 * SUM(correct) / COUNT(*), 1) AS accuracy_pct,
               ROUND(AVG(latency_ms), 0) AS avg_latency_ms
        FROM eval_runs
        GROUP BY strategy
        ORDER BY accuracy_pct DESC
        """
    ).fetchall()
    for r in rows:
        print(f"  {r['strategy']:<10} {r['correct']}/{r['total']} correct "
              f"({r['accuracy_pct']}%)   avg latency {r['avg_latency_ms']:.0f}ms")

    print("\n=== Misclassified cases (by strategy) ===")
    rows = conn.execute(
        """
        SELECT strategy, case_id, input_text, expected_urgency, predicted_urgency
        FROM eval_runs
        WHERE correct = 0
        ORDER BY strategy, case_id
        """
    ).fetchall()
    if not rows:
        print("  None -- every strategy got every case right.")
    for r in rows:
        print(f"  [{r['strategy']}] case {r['case_id']}: \"{r['input_text'][:50]}...\" "
              f"expected={r['expected_urgency']} got={r['predicted_urgency']}")

    print("\n=== Emergency-case recall (most safety-critical) ===")
    rows = conn.execute(
        """
        SELECT strategy,
               SUM(CASE WHEN expected_urgency='Emergency' AND correct=1 THEN 1 ELSE 0 END) AS caught,
               SUM(CASE WHEN expected_urgency='Emergency' THEN 1 ELSE 0 END) AS total_emergencies
        FROM eval_runs
        GROUP BY strategy
        """
    ).fetchall()
    for r in rows:
        print(f"  {r['strategy']:<10} caught {r['caught']}/{r['total_emergencies']} emergency cases")

    conn.close()


if __name__ == "__main__":
    main()
