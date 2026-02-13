# agent/memory.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# -------- Back-compatible defaults (legacy single-store) --------
MEMORY_FILE = Path(__file__).with_name("memory_store.json")

_DEFAULT_MEM = {
    "facts": [],
    "experiences": []
}

MAX_EXPERIENCES = 200   # cap history to keep files small/fast
MAX_TASK_LEN = 300      # avoid huge logs


def _safe_load(path: Path) -> dict:
    if not path.exists():
        return _DEFAULT_MEM.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("facts", [])
        data.setdefault("experiences", [])
        return data
    except Exception:
        return _DEFAULT_MEM.copy()


def _safe_save(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------- Class-based multi-store API (new) --------
class MemoryStore:
    """Isolated memory store per agent/persona."""
    def __init__(self, file_path: Path | str):
        self.path = Path(file_path)

    def load(self) -> dict:
        return _safe_load(self.path)

    def save(self, memory: dict):
        _safe_save(self.path, memory)

    def get_facts(self) -> list[str]:
        return self.load().get("facts", [])

    def remember_fact(self, fact: str) -> bool:
        fact = (fact or "").strip()
        if not fact:
            return False
        mem = self.load()
        facts = mem["facts"]
        if fact.lower() in {f.lower() for f in facts}:
            return False
        facts.append(fact)
        self.save(mem)
        return True

    def remember_experience(self, task: str, outcome: str, meta: Optional[dict] = None):
        mem = self.load()
        exps = mem["experiences"]
        exps.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "task": (task or "")[:MAX_TASK_LEN],
            "outcome": outcome,
            "meta": meta or {}
        })
        # trim
        if len(exps) > MAX_EXPERIENCES:
            mem["experiences"] = exps[-MAX_EXPERIENCES:]
        self.save(mem)

    def clear(self):
        self.save(_DEFAULT_MEM.copy())


# -------- Legacy functions (still work for single-agent mode) --------
def _default_store() -> MemoryStore:
    return MemoryStore(MEMORY_FILE)

def load_memory():
    return _default_store().load()

def save_memory(memory: dict):
    return _default_store().save(memory)

def get_facts() -> list[str]:
    return _default_store().get_facts()

def remember_fact(fact: str) -> bool:
    return _default_store().remember_fact(fact)

def remember_experience(task: str, outcome: str, meta: dict | None = None):
    return _default_store().remember_experience(task, outcome, meta)

def clear_memory():
    return _default_store().clear()
