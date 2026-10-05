"""
Prompt variants for LLM-based symptom triage.
Each function takes a patient's free-text symptom description and returns
the full prompt (system + user) to send to the LLM.
"""

SYSTEM_BASE = (
    "You are a medical symptom triage assistant. Given a patient's free-text "
    "description of their symptoms, classify the urgency and recommend an action.\n\n"
    "Respond ONLY with a JSON object in this exact format, no extra text:\n"
    '{"urgency": "Emergency" | "Urgent" | "Routine", "action": "<one short sentence>"}'
)

FEW_SHOT_EXAMPLES = """
Examples:

Input: "slight itchy rash on arm after gardening, no other symptoms"
Output: {"urgency": "Routine", "action": "Self-care, monitor and see a doctor if it spreads or worsens"}

Input: "severe abdominal pain, vomiting blood"
Output: {"urgency": "Emergency", "action": "Go to ER immediately"}

Input: "mild ankle pain after walking too much, no swelling"
Output: {"urgency": "Routine", "action": "Rest and monitor, see a doctor if pain persists"}
"""

PERSONA_PREFIX = (
    "You are Nurse Priya, a triage nurse with 15 years of ER experience. "
    "You've seen thousands of cases and know exactly how to separate true "
    "emergencies from routine complaints, without over- or under-reacting. "
)


def zero_shot_prompt(symptom_text: str):
    """No examples, no persona -- just the base instruction."""
    return {
        "system": SYSTEM_BASE,
        "user": f'Patient description: "{symptom_text}"',
    }


def few_shot_prompt(symptom_text: str):
    """Base instruction + a handful of worked examples."""
    return {
        "system": SYSTEM_BASE + "\n" + FEW_SHOT_EXAMPLES,
        "user": f'Patient description: "{symptom_text}"',
    }


def persona_prompt(symptom_text: str):
    """Base instruction wrapped in an experienced-nurse persona."""
    return {
        "system": PERSONA_PREFIX + SYSTEM_BASE,
        "user": f'Patient description: "{symptom_text}"',
    }


PROMPT_STRATEGIES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "persona": persona_prompt,
}
