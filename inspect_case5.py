import sqlite3

conn = sqlite3.connect("eval_results.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT strategy, raw_response FROM eval_runs WHERE case_id = 5"
).fetchall()
for r in rows:
    print(f"--- {r['strategy']} ---")
    print(r["raw_response"])
    print()
