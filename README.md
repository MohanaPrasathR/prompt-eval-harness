# Prompt-Evaluation Harness for Symptom Triage

A small evaluation harness that compares three LLM prompting strategies — **zero-shot**,
**few-shot**, and **persona-based** — on a symptom-triage classification task, inspired by
[MediTriage](https://github.com/MohanaPrasathR/E-HOSPITAL). Built to understand how prompt
design affects a conversational AI system's accuracy and safety behavior, with every run
logged to SQLite for structured analysis.

## What it does

1. Takes a set of 10 patient symptom descriptions (`test_cases.json`), each labeled with an
   expected urgency level (`Emergency`, `Urgent`, `Routine`) and recommended action.
2. Sends each case through an LLM (via the [Groq](https://groq.com) API, `openai/gpt-oss-20b`)
   using three different prompt strategies defined in `prompts.py`.
3. Parses the model's JSON response, scores it against the expected label, and logs every
   run — input, strategy, prediction, correctness, latency, and the raw response — to a
   SQLite database (`eval_results.db`).
4. `query_results.py` aggregates the logged runs into accuracy-by-strategy, a list of
   misclassified cases, and emergency-case recall (the most safety-critical metric).

## Key finding: a real bug, not just a benchmark

The first run produced empty or truncated responses on the harder test cases. Inspecting the
raw logged responses (`inspect_case5.py`) showed why: `gpt-oss-20b` is a reasoning model that
spends part of its token budget on internal reasoning before writing the final answer. With
`max_tokens=150`, it sometimes burned the entire budget on reasoning for ambiguous cases and
never got to emit the JSON output. Raising the token budget to 500 fixed it.

## Results (after the fix)

| Strategy  | Accuracy | Emergency recall |
|-----------|----------|-------------------|
| Persona   | 85% (17/20) | 8/8 (100%) |
| Zero-shot | 80% (16/20) | 8/8 (100%) |
| Few-shot  | 80% (16/20) | 8/8 (100%) |

- **Persona-based prompting** ("You are an experienced triage nurse...") outperformed the
  other two strategies.
- **All three strategies caught every true emergency** — the single most important safety
  property for a triage system.
- **All three strategies share the same blind spot**: they consistently under-triage
  borderline "Urgent" cases (a sprained ankle, a 2-week persistent cough) as "Routine"
  rather than escalating them. The system is conservative about missing emergencies but not
  about downgrading moderate cases — a bias worth knowing about before shipping something
  like this in production.

## Project structure

```
prompt-eval-harness/
├── test_cases.json      # 10 labeled symptom-triage test cases
├── prompts.py            # zero-shot / few-shot / persona prompt builders
├── run_eval.py            # runs all cases x strategies through the LLM, logs to SQLite
├── query_results.py       # aggregates results: accuracy, misclassifications, recall
├── inspect_case5.py       # ad-hoc script used to debug the truncation bug
├── list_models.py         # lists models available to the configured API key
├── eval_results.db        # SQLite database of all logged runs
└── .env                   # GROQ_API_KEY (not committed - see .gitignore)
```

## Running it

```bash
pip install groq python-dotenv
# create a .env file with: GROQ_API_KEY=your_key_here
python run_eval.py
python query_results.py
```

## Why this exists

Built as a hands-on extension of [MediTriage](https://github.com/MohanaPrasathR/E-HOSPITAL)
to get real experience with LLM prompt engineering and evaluation — writing and comparing
prompt variants, building a structured test harness, and debugging a genuine model-behavior
issue rather than just reading about prompting techniques.
