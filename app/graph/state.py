from typing import TypedDict, Optional, Any

class TravelState(TypedDict, total=False):
    conversation_id: str
    user_input: str

    # ✅ memory passed in from DB
    convo_context: dict[str, Any]

    intent: str
    slots: dict[str, Any]

    reply: str
    results: dict[str, Any]
    trace: list[dict]
    next_question: Optional[str]

    # ✅ memory to write back to DB
    updated_context: dict[str, Any]
