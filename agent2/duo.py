# agent/duo.py
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
You are a fast, concise assistant.

PERSONA:
{base_persona}

KNOWN FACTS ABOUT THE OTHER PARTY (from your memory):
{known}

RULES:
- Reply in at most 2 short sentences OR 3 bullets.
- Be helpful, specific, and avoid repetition.
- If unsure, say "I don't know".
- Optionally ask at most one clarifying question.
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
            meta={"reply_len": len(reply or "")}
        )
        return reply or "I don't know."

def safe_mem_filename_from(name: str, default_tag: str) -> Path:
    safe = "".join(ch for ch in (name or default_tag) if ch.isalnum() or ch in ("_", "-")).strip() or default_tag
    return Path(f"memory_store_{safe}.json")

def parse_args():
    p = argparse.ArgumentParser(description="Two-agent autonomous chat (auto mode supported).")

    # Personas & names (required for non-interactive auto mode)
    p.add_argument("--a-name", default=None, help="Agent A name")
    p.add_argument("--a-persona", default=None, help="Agent A persona (1-2 sentences)")
    p.add_argument("--b-name", default=None, help="Agent B name")
    p.add_argument("--b-persona", default=None, help="Agent B persona (1-2 sentences)")

    # Memory files (optional; default derived from names)
    p.add_argument("--a-mem", default=None, help="Memory file for Agent A")
    p.add_argument("--b-mem", default=None, help="Memory file for Agent B")

    # Auto mode controls
    p.add_argument("--auto", type=int, default=0, help="Run N turns automatically (no interaction)")
    p.add_argument("--seed", default="Hello", help="Seed message for the first speaker")
    p.add_argument("--starter", choices=["A", "B"], default="A", help="Who starts the conversation")
    p.add_argument("--delay", type=float, default=0.0, help="Delay (seconds) between printed messages")

    # Transcript
    p.add_argument("--transcript", default=None, help="Path to save conversation transcript")

    return p.parse_args()

def ensure_names_personas(args) -> tuple[Agent, Agent]:
    # If any missing, fill with reasonable defaults (keeps it non-interactive)
    a_name = args.a_name or "Aria"
    b_name = args.b_name or "Blaze"
    a_persona = args.a_persona or "A pragmatic senior software architect who prefers concise, actionable answers."
    b_persona = args.b_persona or "A curious product manager focused on users, clarity, and scope."

    a_mem = Path(args.a_mem) if args.a_mem else safe_mem_filename_from(a_name, "A")
    b_mem = Path(args.b_mem) if args.b_mem else safe_mem_filename_from(b_name, "B")

    agentA = Agent(a_name, a_persona, a_mem)
    agentB = Agent(b_name, b_persona, b_mem)
    return agentA, agentB

def print_header(a: Agent, b: Agent, auto_turns: int, seed: str, starter: str, delay: float, transcript_path: str | None):
    print("✅ Two-Agent runner")
    print(f"  A: {a.name}  (mem: {a.mem.path.name})")
    print(f"  B: {b.name}  (mem: {b.mem.path.name})")
    if auto_turns > 0:
        print(f"  Mode: AUTO for {auto_turns} turns | starter: {starter} | seed: {seed!r} | delay: {delay}s")
    else:
        print("  Mode: Interactive (type '/quit' to exit)")
    if transcript_path:
        print(f"  Transcript: {transcript_path}")
    print()

def write_line(fp, line: str):
    if fp:
        fp.write(line + "\n")
        fp.flush()

def auto_run(a: Agent, b: Agent, seed: str, turns: int, starter: str, delay: float, transcript_fp=None):
    """
    Alternate between A and B for N turns, starting with `starter`.
    One 'turn' = one message by the current speaker and one reply by the other.
    """
    speaker_is_A = (starter.upper() == "A")
    msg = seed

    for i in range(turns):
        if speaker_is_A:
            # A speaks, B replies
            line = f"A ({a.name}): {msg}"
            print(line); write_line(transcript_fp, line)
            reply = b.respond(msg, from_name=a.name)
            line = f"B ({b.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            msg = reply
        else:
            # B speaks, A replies
            line = f"B ({b.name}): {msg}"
            print(line); write_line(transcript_fp, line)
            reply = a.respond(msg, from_name=b.name)
            line = f"A ({a.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            msg = reply

        # alternate speaker; optional pacing
        speaker_is_A = not speaker_is_A
        if delay > 0:
            time.sleep(delay)

def interactive_loop(a: Agent, b: Agent, transcript_fp=None):
    print("Commands: 'A: <msg>' | 'B: <msg>' | '/quit'")
    last_speaker = "B"  # so A starts if you just type a bare line

    while True:
        raw = input("\n> ").strip()
        if not raw:
            continue
        if raw.lower() in ("/q", "/quit", "exit"):
            break

        if raw.startswith("A:"):
            msg = raw[2:].strip()
            line = f"A ({a.name}): {msg}"
            print(line); write_line(transcript_fp, line)
            reply = b.respond(msg, from_name=a.name)
            line = f"B ({b.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            last_speaker = "A"
            continue

        if raw.startswith("B:"):
            msg = raw[2:].strip()
            line = f"B ({b.name}): {msg}"
            print(line); write_line(transcript_fp, line)
            reply = a.respond(msg, from_name=b.name)
            line = f"A ({a.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            last_speaker = "B"
            continue

        # bare line: alternate automatically
        if last_speaker == "A":
            line = f"B ({b.name}): {raw}"
            print(line); write_line(transcript_fp, line)
            reply = a.respond(raw, from_name=b.name)
            line = f"A ({a.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            last_speaker = "B"
        else:
            line = f"A ({a.name}): {raw}"
            print(line); write_line(transcript_fp, line)
            reply = b.respond(raw, from_name=a.name)
            line = f"B ({b.name}): {reply}"
            print(line); write_line(transcript_fp, line)
            last_speaker = "A"

def main():
    args = parse_args()
    agentA, agentB = ensure_names_personas(args)

    transcript_fp = open(args.transcript, "a", encoding="utf-8") if args.transcript else None
    try:
        print_header(agentA, agentB, args.auto, args.seed, args.starter, args.delay, args.transcript)
        if args.auto and args.auto > 0:
            auto_run(agentA, agentB, args.seed, args.auto, args.starter, args.delay, transcript_fp)
        else:
            interactive_loop(agentA, agentB, transcript_fp)
    finally:
        if transcript_fp:
            transcript_fp.close()

if __name__ == "__main__":
    main()