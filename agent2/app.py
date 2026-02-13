# agent/app.py
import re
from memory import (
    get_facts,
    remember_fact,
    remember_experience,
    clear_memory,
)
from llm import call_llm

# ---------- BLOCKED INPUTS ----------
BLOCKED_INPUTS = {
    "sex", "porn", "xxx"
}

# ---------- FACT EXTRACTION ----------
PREFERRED_PATTERNS = [
    re.compile(r"\bI\s+prefer\s+([^.]+)", re.I),
    re.compile(r"\bI\s+like\s+([^.]+)", re.I),
]

IDENTITY_PATTERNS = [
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z .'-]+)", re.I),
]

def extract_candidate_facts(text: str) -> list[str]:
    facts = []

    for pat in PREFERRED_PATTERNS:
        for m in pat.finditer(text):
            facts.append(f"User prefers {m.group(1).strip()}")

    for pat in IDENTITY_PATTERNS:
        for m in pat.finditer(text):
            name = " ".join(w.capitalize() for w in m.group(1).split())
            facts.append(f"User name is {name}")

    # dedupe
    return list(dict.fromkeys(facts))

# ---------- FAST MEMORY CHECK ----------
def is_memory_only(text: str, facts: list[str]) -> bool:
    if not facts:
        return False
    if "?" in text:
        return False
    keywords = ("how", "what", "why", "when", "where", "help", "suggest")
    return not any(k in text.lower() for k in keywords)

# ---------- SYSTEM PROMPT ----------
def build_system_prompt(facts: list[str]) -> str:
    known = "\n".join(f"- {f}" for f in facts) or "- (none)"
    return f"""
You are a fast engineering assistant.

Known user facts:
{known}

RULES:
- Max 2 sentences OR 3 bullets
- No repetition
- If unsure say "I don't know"
""".strip()

# ---------- AGENT LOOP ----------
def run_agent_loop():
    print("✅ Fast AI Agent started (type 'exit')")

    while True:
        user_input = input("\nUser > ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.lower() == "memory clear":
            clear_memory()
            print("✅ Memory cleared")
            continue

        if user_input.lower() in BLOCKED_INPUTS:
            print("Agent > Let's change the topic.")
            continue

        # 1️⃣ Extract memory
        new_facts = extract_candidate_facts(user_input)
        added = [f for f in new_facts if remember_fact(f)]

        # 2️⃣ Memory-only fast path
        if is_memory_only(user_input, new_facts):
            print("Agent > Got it.")
            remember_experience(user_input, "memory-only", {"facts": added})
            continue

        # 3️⃣ LLM call
        system_prompt = build_system_prompt(get_facts())
        response = call_llm(user_input, system_prompt)

        print("Agent >", response)

        remember_experience(user_input, "ok", {"facts": added})

if __name__ == "__main__":
    run_agent_loop()