# app/llm/dialogue_manager.py
import os
import json
import re
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


SYSTEM_PROMPT = """
You are the Dialogue Manager for an agentic travel planner.

You decide the NEXT best action based on:
- user_input
- memory slots
- memory results
- conversation history
- last_reply
- last_question_key (if provided)

You must output ONLY valid JSON (no markdown, no explanations).

Allowed actions:
- "flights"
- "hotels"
- "cabs"
- "covid"
- "ask_user"

Slots you may produce (when relevant):
Flights:
- origin (city)
- destination (city)
- date (YYYY-MM-DD)

Hotels:
- city
- checkin (YYYY-MM-DD)
- checkout (YYYY-MM-DD)

Cabs:
- pickup
- dropoff

Covid:
- country (preferred) OR destination

Rules:
1) If missing critical info for an action, choose action "ask_user" and provide ask_user question.
2) If the user replies yes/no, infer meaning from last_question_key (preferred) else last_reply/history.
3) IMPORTANT FOLLOW-UP:
   After a successful flight search (state.results has flights),
   the assistant should ask:
   "Do you want to book a hotel OR see COVID updates for your destination country?"
   The user can reply: "hotel" / "covid" / "yes" / "no".
   - "hotel" or "yes" -> action "hotels"
   - "covid" -> action "covid"
   - "no" -> action "ask_user" (ask what next)
4) If user explicitly says "covid" / "cases" / "covid update" -> action "covid".
5) Keep questions short and actionable.

Output JSON schema:
{
  "action": "flights|hotels|cabs|covid|ask_user",
  "slots": { ... },
  "missing": ["field1","field2"],
  "ask_user": "question string if action=ask_user else null"
}
"""


def _safe_json_parse(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def _normalize_text(t: str) -> str:
    return (t or "").strip().lower()


def _extract_yes_no(t: str) -> Optional[str]:
    """
    Returns "yes" / "no" or None.
    """
    s = _normalize_text(t)
    # common short replies
    if s in {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "haan", "ha", "ji", "please do"}:
        return "yes"
    if s in {"n", "no", "nope", "nah", "not now", "dont", "don't"}:
        return "no"
    # if sentence contains explicit yes/no intent
    if re.search(r"\b(yes|yeah|yep|sure|okay|ok)\b", s):
        return "yes"
    if re.search(r"\b(no|nope|nah)\b", s):
        return "no"
    return None


def _extract_choice_hotel_or_covid(t: str) -> Optional[str]:
    """
    Returns "hotels" / "covid" / None.
    """
    s = _normalize_text(t)
    if re.search(r"\b(hotel|hotels)\b", s):
        return "hotels"
    if re.search(r"\b(covid|cases|case count|covid update|covid updates|corona)\b", s):
        return "covid"
    return None


def _has_flights(results: Dict[str, Any]) -> bool:
    flights = (results or {}).get("flights") or {}
    arr = flights.get("flights") or []
    return isinstance(arr, list) and len(arr) > 0


_llm = ChatOpenAI(model=MODEL, temperature=0)


def plan_next_step(user_input: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid planner:
    - deterministic routing for the specific follow-up:
      "hotel OR covid?" after flights
    - LLM handles everything else
    """
    state = state or {}
    results = state.get("results") or {}
    last_reply = state.get("last_reply") or ""
    last_key = state.get("last_question_key")  # may be None
    history = state.get("history") or []

    txt = user_input or ""
    yn = _extract_yes_no(txt)
    choice = _extract_choice_hotel_or_covid(txt)

    # 1) Deterministic handling of the "after flights" follow-up (best reliability)
    # Prefer last_question_key if you set it (recommended).
    if last_key == "AFTER_FLIGHT_NEXT_STEP":
        if choice == "covid":
            return {"action": "covid", "slots": {}, "missing": [], "ask_user": None}
        if choice == "hotels":
            return {"action": "hotels", "slots": {}, "missing": [], "ask_user": None}
        if yn == "yes":
            # yes defaults to hotel (as per your requirement)
            return {"action": "hotels", "slots": {}, "missing": [], "ask_user": None}
        if yn == "no":
            return {
                "action": "ask_user",
                "slots": {},
                "missing": [],
                "ask_user": "Okay — what would you like to do next (flights / hotels / cabs / covid)?",
            }

        # unclear reply, ask again
        return {
            "action": "ask_user",
            "slots": {},
            "missing": [],
            "ask_user": "Do you want to book a hotel or see COVID updates? Reply: hotel / covid / no",
        }

    # 2) If flights exist and user asks covid/hotel explicitly, route fast
    if _has_flights(results) and choice in {"hotels", "covid"}:
        return {"action": choice, "slots": {}, "missing": [], "ask_user": None}

    # 3) Otherwise call LLM for intent + slot extraction
    msg = HumanMessage(
        content=json.dumps(
            {"user_input": user_input, "state": state},
            ensure_ascii=False,
        )
    )
    resp = _llm.invoke([SystemMessage(content=SYSTEM_PROMPT), msg])
    plan = _safe_json_parse(resp.content)

    action = plan.get("action") or "ask_user"
    if action not in {"flights", "hotels", "cabs", "covid", "ask_user"}:
        action = "ask_user"

    return {
        "action": action,
        "slots": plan.get("slots") or {},
        "missing": plan.get("missing") or [],
        "ask_user": plan.get("ask_user"),
    }
