from langgraph.graph import StateGraph, END

from app.graph.state import TravelState
from app.llm.dialogue_manager import plan_next_step

# Providers
from app.providers.amadeus_flights import AmadeusFlightsProvider
from app.providers.amadeus_hotels import AmadeusHotelsProvider
from app.providers.mock_cabs import MockCabsProvider

# Agents
from app.agents.hotels import run_hotels_agent
from app.agents.cabs import run_cabs_agent


# ---------------------------
# Providers init
# ---------------------------
flights_provider = AmadeusFlightsProvider()
hotels_provider = AmadeusHotelsProvider()
cabs_provider = MockCabsProvider()


# ---------------------------
# Utilities
# ---------------------------
def add_trace(state: TravelState, node: str, detail: dict):
    state.setdefault("trace", [])
    state["trace"].append({"node": node, "detail": detail})


def _ctx_history_append(ctx: dict, role: str, content: str):
    hist = ctx.get("history") or []
    hist.append({"role": role, "content": content})
    ctx["history"] = hist[-30:]  # keep last 30 messages


def _persist_context(state: TravelState, ctx: dict) -> TravelState:
    """
    Always write back slots/results/history so the LLM has memory.
    """
    ctx = dict(ctx or {})
    ctx["slots"] = state.get("slots", {}) or {}
    ctx["results"] = state.get("results", {}) or {}
    state["updated_context"] = ctx
    return state


# ---------------------------
# LLM Master Node
# ---------------------------
def node_master_llm(state: TravelState) -> TravelState:
    user_text = (state.get("user_input") or "").strip()
    ctx = state.get("convo_context", {}) or {}

    # load memory
    slots = ctx.get("slots") or {}
    results = ctx.get("results") or {}
    history = ctx.get("history") or []

    # record user message into history for planning
    ctx2 = dict(ctx)
    _ctx_history_append(ctx2, "user", user_text)

    plan = plan_next_step(
        user_input=user_text,
        state={
            "slots": slots,
            "results": results,
            "history": history,
            "last_reply": ctx.get("last_reply"),
        }
    )

    state["llm_action"] = plan["action"]
    state["llm_missing"] = plan.get("missing", []) or []

    # merge existing slots with new slots from LLM (so we keep memory)
    new_slots = dict(slots)
    new_slots.update(plan.get("slots") or {})
    state["slots"] = new_slots

    add_trace(state, "llm_plan", plan)

    if plan["action"] == "ask_user":
        q = plan.get("ask_user") or "Can you clarify?"
        state["reply"] = q
        state["next_question"] = q

        # record assistant question into history
        _ctx_history_append(ctx2, "assistant", q)
        ctx2["last_reply"] = q
        state = _persist_context(state, ctx2)
        return state

    # no direct reply here; tool nodes will reply
    state = _persist_context(state, ctx2)
    return state


def node_route(state: TravelState) -> str:
    return state.get("llm_action", "ask_user")


# ---------------------------
# Tool Nodes
# ---------------------------
def node_flights(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    required = ["origin", "destination", "date"]
    missing = [k for k in required if not s.get(k)]
    if missing:
        q = f"I can search flights, but I still need: {', '.join(missing)}."
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "flights_missing", {"missing": missing})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    # Local import to avoid NameError
    from app.agents.flights import run_flights_agent

    try:
        data = run_flights_agent(flights_provider, s["origin"], s["destination"], s["date"])
    except Exception as e:
        q = f"Flight search failed: {str(e)}. Try a different date/city."
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "flights_error", {"error": str(e)})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    state.setdefault("results", {})
    state["results"]["flights"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        q = f"No flights found for {s['origin']} → {s['destination']} on {s['date']}. Try another date?"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "flights_none", {})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    reply = (
        f"Here are flights from {s['origin']} to {s['destination']} on {s['date']} (cheapest first).\n"
        f"Cheapest: {cheapest.get('airline')} {cheapest.get('flight_no')} ₹{cheapest.get('price_inr')} "
        f"({cheapest.get('depart')} → {cheapest.get('arrive')}).\n\n"
        "Do you want to book a hotel as well?"
    )
    state["reply"] = reply
    state["next_question"] = "Do you want to book a hotel as well?"
    add_trace(state, "flights_ok", {"cheapest": cheapest})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    return _persist_context(state, ctx)


def node_hotels(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    required = ["city", "checkin", "checkout"]
    missing = [k for k in required if not s.get(k)]
    if missing:
        # ask only one thing at a time
        if "checkout" in missing and s.get("city") and s.get("checkin"):
            q = "Please enter last day of your stay (check-out date)."
        elif "checkin" in missing and s.get("city"):
            q = "What is your check-in date?"
        elif "city" in missing:
            q = "Which city would you like to book the hotel in?"
        else:
            q = f"I need: {', '.join(missing)} to search hotels."

        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "hotels_missing", {"missing": missing})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    try:
        data = run_hotels_agent(hotels_provider, s["city"], s["checkin"], s["checkout"])
    except Exception as e:
        q = f"Hotel search failed: {str(e)}. Try another city (DEL/BOM) or dates."
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "hotels_error", {"error": str(e)})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    state.setdefault("results", {})
    state["results"]["hotels"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        q = f"No hotels found in {s['city']} for {s['checkin']} → {s['checkout']}. Try different dates?"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "hotels_none", {})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    name = cheapest.get("name", "Unknown Hotel")
    per_night = cheapest.get("price_per_night_inr")
    total = cheapest.get("price_total_inr")
    rating = cheapest.get("rating")

    rating_txt = f" (⭐ {rating})" if rating else ""
    price_txt = ""
    if per_night is not None:
        price_txt = f"₹{per_night}/night"
        if total is not None:
            price_txt += f" (₹{total} total)"
    elif total is not None:
        price_txt = f"₹{total} total"
    else:
        price_txt = "Price unavailable"

    reply = (
        f"Here are hotels in {s['city']} from {s['checkin']} to {s['checkout']} (cheapest first).\n"
        f"Cheapest: {name} {price_txt}{rating_txt}.\n\n"
        "Do you want to book a cab from your destination (airport) to your selected hotel?"
    )
    state["reply"] = reply
    state["next_question"] = "Do you want to book a cab from your destination (airport) to your selected hotel?"
    add_trace(state, "hotels_ok", {"cheapest": cheapest})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    return _persist_context(state, ctx)


def node_cabs(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    required = ["pickup", "dropoff"]
    missing = [k for k in required if not s.get(k)]
    if missing:
        # Ask for hotel selection if pickup known but dropoff missing
        if "dropoff" in missing and s.get("pickup"):
            q = "Which hotel did you select? (Type the hotel name)"
        elif "pickup" in missing:
            q = "Where should the cab pick you up from? (e.g., Mumbai Airport)"
        else:
            q = f"I need: {', '.join(missing)} to search cabs."

        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_missing", {"missing": missing})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    try:
        data = run_cabs_agent(cabs_provider, s["pickup"], s["dropoff"])
    except Exception as e:
        q = f"Cab search failed: {str(e)}"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_error", {"error": str(e)})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    state.setdefault("results", {})
    state["results"]["cabs"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        q = f"No cabs found for {s['pickup']} → {s['dropoff']}. Try changing pickup/dropoff?"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_none", {})

        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    reply = (
        f"Cab options from {s['pickup']} to {s['dropoff']} (cheapest first).\n"
        f"Cheapest: {cheapest.get('vendor')} {cheapest.get('type')} ₹{cheapest.get('fare_inr')} "
        f"(ETA {cheapest.get('eta_min')} min).\n\n"
        "Anything else you want to add (flights/hotels)?"
    )
    state["reply"] = reply
    state["next_question"] = "Anything else you want to add?"
    add_trace(state, "cabs_ok", {"cheapest": cheapest})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    return _persist_context(state, ctx)


# ---------------------------
# Build graph
# ---------------------------
def build_graph():
    g = StateGraph(TravelState)

    g.add_node("master", node_master_llm)
    g.add_node("flights", node_flights)
    g.add_node("hotels", node_hotels)
    g.add_node("cabs", node_cabs)

    g.set_entry_point("master")

    g.add_conditional_edges("master", node_route, {
        "flights": "flights",
        "hotels": "hotels",
        "cabs": "cabs",
        "ask_user": END,
    })

    g.add_edge("flights", END)
    g.add_edge("hotels", END)
    g.add_edge("cabs", END)

    return g.compile()
