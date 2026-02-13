# agent/duo_auto.py
import argparse
import time
import re
from pathlib import Path

from llm import call_llm
from memory import MemoryStore

# ---------- Config ----------
BLOCKED_INPUTS = {"sex", "porn", "xxx"}  # keep logs clean/safe

PREFERRED_PATTERNS = [
    re.compile(r"\bI\s+prefer\s+([^.?!\n]+)", re.I),
    re.compile(r"\bI\s+like\s+([^.?!\n]+)", re.I),
]
IDENTITY_PATTERNS = [
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z .'-]+)", re.I),
]
COLOR_PATTERN = re.compile(r"\bmy\s+favou?rite\s+color\s+is\s+([a-zA-Z]+)\b", re.I)

def extract_candidate_facts(text: str) -> list[str]:
    """Deterministic, small-footprint memory extraction (no LLM)."""
    facts: list[str] = []
    for pat in PREFERRED_PATTERNS:
        for m in pat.finditer(text):
            facts.append(f"User prefers {m.group(1).strip()}")
    for m in IDENTITY_PATTERNS:
        for g in m.finditer(text):
            name = " ".join(w.capitalize() for w in g.group(1).split())
            facts.append(f"User name is {name}")
    for m in COLOR_PATTERN.finditer(text):
        facts.append(f"User's favorite color is {m.group(1).strip().lower()}")

    # dedupe + cleanup
    cleaned, seen = [], set()
    for f in facts:
        ff = f.rstrip(" .!?,;").strip()
        lf = ff.lower()
        if ff and lf not in seen:
            cleaned.append(ff)
            seen.add(lf)
    return cleaned


def build_system_prompt(base_persona: str, known_facts: list[str]) -> str:
    known = "\n".join(f"- {f}" for f in known_facts) if known_facts else "- (no saved facts yet)"
    return f"""
You are chatting in an ONGOING casual conversation.

PERSONA:
{base_persona}

KNOWN CONTEXT (from memory):
{known}

STRICT CHAT RULES:
- This is NOT an email or letter.
- Do NOT greet with "Hi", "Hello", or the person's name.
- Do NOT say goodbye, bye, take care, or sign your name.
- Do NOT restart the conversation.
- Speak like a real person chatting.
- Use 1–2 short sentences only.
- No bullet points unless necessary.
- Emojis are allowed but minimal (😊 😄).

Just continue the conversation naturally.
""".strip()

class Agent:
    def __init__(self, name: str, persona: str, memory_file: Path):
        self.name = name
        self.persona = persona
        self.mem = MemoryStore(memory_file)

    def _remember_from_input(self, text: str):
        for fact in extract_candidate_facts(text):
            self.mem.remember_fact(fact)

    def respond(self, incoming_text: str, from_name: str) -> str:
        # safety block
        if incoming_text.lower().strip() in BLOCKED_INPUTS:
            reply = "Let's change the topic."
            self.mem.remember_experience(
                task=f"Blocked input received from {from_name}",
                outcome="blocked",
                meta={}
            )
            return reply

        # learn from what the other said
        self._remember_from_input(incoming_text)

        # craft system prompt using THIS agent's memory
        system_prompt = build_system_prompt(
            base_persona=f"You are {self.name}. {self.persona}",
            known_facts=self.mem.get_facts()
        )

        # produce reply
        reply = call_llm(
            user_prompt=f"{from_name} said: {incoming_text}\nReply as {self.name}.",
            system_prompt=system_prompt
        )

        # log experience
        self.mem.remember_experience(
            task=f"From {from_name} -> {self.name}: {incoming_text}",
            outcome="ok" if reply else "empty-response",
            meta={"reply_len": len((reply or '').strip())}
        )
        return (reply or "I don't know.").strip()

def safe_mem_filename_from(name: str, default_tag: str) -> Path:
    safe = "".join(ch for ch in (name or default_tag) if ch.isalnum() or ch in ("_", "-")).strip() or default_tag
    return Path(f"memory_store_{safe}.json")

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-agent autonomous chat (non-interactive). This script never asks for input."
    )
    # Required: names & personas (to avoid any prompt)
    p.add_argument("--a-name", required=True, help="Agent A name (required)")
    p.add_argument("--a-persona", required=True, help="Agent A persona (1-2 sentences, required)")
    p.add_argument("--b-name", required=True, help="Agent B name (required)")
    p.add_argument("--b-persona", required=True, help="Agent B persona (1-2 sentences, required)")

    # Optional: memory files (default derived from names)
    p.add_argument("--a-mem", default=None, help="Memory file for Agent A")
    p.add_argument("--b-mem", default=None, help="Memory file for Agent B")

    # Auto mode controls
    p.add_argument("--turns", type=int, default=20, help="Number of alternating turns (default: 20)")
    p.add_argument("--starter", choices=["A", "B"], default="A", help="Which agent starts (default: A)")
    p.add_argument("--seed", default="Hello", help="Seed message for the starter (default: 'Hello')")
    p.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between message pairs (default: 0.0)")

    # Transcript
    p.add_argument("--transcript", default=None, help="Save transcript to this file (optional)")

    return p.parse_args()

def write_line(fp, line: str):
    if fp:
        fp.write(line + "\n")
        fp.flush()

def auto_run(a: Agent, b: Agent, seed: str, turns: int, starter: str, delay: float, transcript_fp=None):
    """
    Alternate A and B for `turns` *single* messages (no duplicates).
    We only print a message when that agent is the current speaker.
    The reply is stored and becomes the next speaker's outgoing message,
    but is NOT printed immediately.
    """
    # Who will speak first?
    speaker_is_A = (starter.upper() == "A")

    # Bootstrap: the starting message (seed) is what the starter speaks first.
    pending_msg = seed          # the next message to be spoken
    pending_from = "A" if speaker_is_A else "B"  # who will speak the pending message

    for _ in range(turns):
        if pending_from == "A":
            # Print A's outgoing message
            line = f"A ({a.name}): {pending_msg}"
            print(line)
            if transcript_fp:
                transcript_fp.write(line + "\n")

            # Get B's reply (but DO NOT print it now)
            reply = b.respond(pending_msg, from_name=a.name)

            # Next turn: B will "say" that reply
            pending_msg = reply
            pending_from = "B"

        else:  # pending_from == "B"
            line = f"B ({b.name}): {pending_msg}"
            print(line)
            if transcript_fp:
                transcript_fp.write(line + "\n")

            reply = a.respond(pending_msg, from_name=b.name)

            pending_msg = reply
            pending_from = "A"

        if delay > 0:
            import time
            time.sleep(delay)

def main():
    args = parse_args()

    a_mem = Path(args.a_mem) if args.a_mem else safe_mem_filename_from(args.a_name, "A")
    b_mem = Path(args.b_mem) if args.b_mem else safe_mem_filename_from(args.b_name, "B")

    agentA = Agent(args.a_name, args.a_persona, a_mem)
    agentB = Agent(args.b_name, args.b_persona, b_mem)

    # header
    print("✅ Two-Agent AUTO runner (no prompts)")
    print(f"  A: {agentA.name}  (mem: {agentA.mem.path.name})")
    print(f"  B: {agentB.name}  (mem: {agentB.mem.path.name})")
    print(f"  Turns: {args.turns} | Starter: {args.starter} | Seed: {args.seed!r} | Delay: {args.delay}s")
    if args.transcript:
        print(f"  Transcript: {args.transcript}")
    print()

    transcript_fp = open(args.transcript, "a", encoding="utf-8") if args.transcript else None
    try:
        auto_run(agentA, agentB, args.seed, args.turns, args.starter, args.delay, transcript_fp)
    finally:
        if transcript_fp:
            transcript_fp.close()

if __name__ == "__main__":
    main()
