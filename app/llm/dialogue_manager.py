import os
import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
You are the Dialogue Manager for an agentic travel planner.

You must decide the NEXT best action based on the user's message and the current conversation state.

You must output ONLY valid JSON (no markdown, no extra text).

Allowed actions:
- "flights"
- "hotels"
- "cabs"
- "ask_user"

You must produce ISO dates in slots when possible:
- date: YYYY-MM-DD
- checkin/checkout: YYYY-MM-DD

Slot expectations:
Flights slots:
- origin, destination, date

Hotels slots:
- city, checkin, checkout

Cabs slots:
- pickup, dropoff

Rules:
1) If required slots for the intended action are missing, choose "ask_user" and ask ONE clear question.
2) Use the conversation state (known slots/results/history) to infer missing details when safe.
   Example: after flights Delhi->Mumbai on 2026-01-15, if user says "yes book hotel",
   set hotels.city=Mumbai and hotels.checkin=2026-01-15 but ask for checkout if missing.
3) Natural language yes/no should be interpreted based on prior assistant question in history.
4) Prefer continuing the current plan (e.g., flights -> hotels -> cabs) if user agrees.

Output JSON schema:
{
  "action": "flights|hotels|cabs|ask_user",
  "slots": { ... },        // only relevant slots for that action
  "missing": ["..."],      // list of missing required fields (if any)
  "ask_user": "..."        // question ONLY when action is ask_user; else null
}
"""

_llm = ChatOpenAI(model=MODEL, temperature=0.1)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # fallback: minimal safe response
        return {
            "action": "ask_user",
            "slots": {},
            "missing": [],
            "ask_user": "Sorry — I didn’t understand. Can you rephrase your request?"
        }


def plan_next_step(user_input: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state should include:
      - slots
      - results (optional)
      - history (optional)
      - last_reply (optional)
    """
    payload = {
        "known_slots": state.get("slots", {}),
        "known_results_keys": list((state.get("results") or {}).keys()),
        "history": state.get("history", [])[-10:],  # last 10 messages max
        "last_reply": state.get("last_reply"),
        "user_input": user_input
    }

    msg = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    resp = _llm.invoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    plan = _safe_json_parse(resp.content)

    # harden plan structure
    action = plan.get("action") or "ask_user"
    if action not in {"flights", "hotels", "cabs", "ask_user"}:
        action = "ask_user"

    return {
        "action": action,
        "slots": plan.get("slots") or {},
        "missing": plan.get("missing") or [],
        "ask_user": plan.get("ask_user"),
    }
