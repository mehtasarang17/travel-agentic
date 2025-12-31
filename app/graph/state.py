from typing import TypedDict, Optional, Any

class TravelState(TypedDict, total=False):
    conversation_id: str
    user_input: str

    # memory loaded from DB (Conversation.context)
    convo_context: dict[str, Any]

    # LLM plan
    llm_action: str                 # flights|hotels|cabs|ask_user
    llm_missing: list[str]
    slots: dict[str, Any]           # global working slots (persist across turns)

    # outputs
    reply: str
    next_question: Optional[str]
    results: dict[str, Any]
    trace: list[dict]

    # memory to write back to DB
    updated_context: dict[str, Any]
