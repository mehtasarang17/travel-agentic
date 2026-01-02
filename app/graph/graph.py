# app/graph/graph.py
import pycountry
from langgraph.graph import StateGraph, END

from app.graph.state import TravelState
from app.llm.dialogue_manager import plan_next_step

# Providers
from app.providers.amadeus_flights import (
    AmadeusFlightsProvider,
    UnknownLocationError as FlightsUnknownLocationError,
)
from app.providers.amadeus_hotels import (
    AmadeusHotelsProvider,
    UnknownLocationError as HotelsUnknownLocationError,
)
from app.providers.amadeus_cabs import (
    AmadeusCabsProvider,
    UnknownLocationError as CabsUnknownLocationError,
)
from app.providers.disease_covid import DiseaseCovidProvider

# Agents
from app.agents.hotels import run_hotels_agent
from app.agents.cabs import run_cabs_agent
from app.agents.covid import run_covid_agent


# ---------------------------
# Providers init
# ---------------------------
flights_provider = AmadeusFlightsProvider()
hotels_provider = AmadeusHotelsProvider()
cabs_provider = AmadeusCabsProvider()
covid_provider = DiseaseCovidProvider()


# ---------------------------
# Utilities
# ---------------------------
def add_trace(state: TravelState, node: str, detail: dict):
    state.setdefault("trace", [])
    state["trace"].append({"node": node, "detail": detail})


def _ctx_history_append(ctx: dict, role: str, content: str):
    hist = ctx.get("history") or []
    hist.append({"role": role, "content": content})
    ctx["history"] = hist[-30:]


def _persist_context(state: TravelState, ctx: dict) -> TravelState:
    """
    Always write back slots/results/history so the LLM has memory.
    """
    ctx = dict(ctx or {})
    ctx["slots"] = state.get("slots", {}) or {}
    ctx["results"] = state.get("results", {}) or {}
    state["updated_context"] = ctx
    return state


def _looks_like_yes(txt: str) -> bool:
    t = (txt or "").strip().lower()
    return t in {"yes", "y", "sure", "ok", "okay", "haan", "yeah", "yep"}

def _looks_like_no(txt: str) -> bool:
    t = (txt or "").strip().lower()
    return t in {"no", "n", "nope", "nah"}

def _wants_covid(txt: str) -> bool:
    t = (txt or "").strip().lower()
    return ("covid" in t) or ("cases" in t) or ("corona" in t)

def _wants_hotel(txt: str) -> bool:
    t = (txt or "").strip().lower()
    return ("hotel" in t) or _looks_like_yes(t)


def _resolve_country_from_destination(state: TravelState) -> None:
    """
    Fill country info using Amadeus destination lookup so we NEVER ask user for country.
    Writes slots:
      - destination_country_code (ISO2)
      - destination_country (country name)
      - country (country name)  <-- critical for covid flow
    """
    slots = state.get("slots", {}) or {}
    dest = (slots.get("destination") or "").strip()
    if not dest:
        dest = (slots.get("destination_iata") or "").strip()

    if not dest:
        add_trace(state, "dest_country_skip", {"reason": "missing destination"})
        return

    try:
        loc = flights_provider.resolve_location_country(dest)  # must use SDK implementation
        cc = (loc or {}).get("countryCode")
        if not cc:
            add_trace(state, "dest_country_not_found", {"destination": dest, "loc": loc})
            return

        country_name = None
        try:
            c = pycountry.countries.get(alpha_2=cc.upper())
            country_name = c.name if c else None
        except Exception:
            country_name = None

        # write back to slots
        slots["destination_country_code"] = cc.upper()
        if country_name:
            slots["destination_country"] = country_name
            slots["country"] = country_name   # ✅ IMPORTANT: what your covid node expects
        else:
            # fallback: at least set country as code so covid node can try
            slots["country"] = cc.upper()

        state["slots"] = slots
        add_trace(state, "dest_country_resolved", {
            "destination": dest,
            "countryCode": slots.get("destination_country_code"),
            "country": slots.get("country"),
        })

    except Exception as e:
        add_trace(state, "dest_country_resolve_failed", {"destination": dest, "error": str(e)})


# ---------------------------
# LLM Master Node
# ---------------------------
def node_master_llm(state: TravelState) -> TravelState:
    user_text = (state.get("user_input") or "").strip()
    ctx = state.get("convo_context", {}) or {}

    # Load memory
    slots = ctx.get("slots") or {}
    results = ctx.get("results") or {}
    history = ctx.get("history") or []
    last_q = ctx.get("last_question_key")
    last_reply = ctx.get("last_reply")

    # Record user msg
    ctx2 = dict(ctx)
    _ctx_history_append(ctx2, "user", user_text)

    # ✅ Deterministic routing for the "Hotel or COVID?" follow-up
    # This avoids LLM guessing and ensures smooth flow.
    if last_q == "AFTER_FLIGHT_NEXT_STEP":
        # start from memory slots
        merged = dict(slots)

        if _wants_covid(user_text):
            # ensure country is filled from amadeus (if not already)
            state["slots"] = merged
            _resolve_country_from_destination(state)
            merged = state.get("slots") or merged

            state["llm_action"] = "covid"
            state["llm_missing"] = []
            state["slots"] = merged
            add_trace(state, "master_after_flight_route", {"choice": "covid", "slots": merged})
            add_trace(state, "master_slots_seen", {"slots": ctx.get("slots")})

            return _persist_context(state, ctx2)

        if _wants_hotel(user_text):
            state["llm_action"] = "hotels"
            state["llm_missing"] = []
            # seed hotel slots from flight info
            merged.setdefault("city", merged.get("destination"))
            merged.setdefault("checkin", merged.get("date"))
            state["slots"] = merged
            add_trace(state, "master_after_flight_route", {"choice": "hotels", "slots": merged})
            return _persist_context(state, ctx2)

        if _looks_like_no(user_text):
            q = "Okay 👍 Anything else you’d like to search (flights/hotels/cabs/covid)?"
            state["reply"] = q
            state["next_question"] = q
            _ctx_history_append(ctx2, "assistant", q)
            ctx2["last_reply"] = q
            ctx2["last_question_key"] = "OPEN_ENDED"
            return _persist_context(state, ctx2)

        # If unclear
        q = "Please reply with: hotel / covid / or no."
        state["reply"] = q
        state["next_question"] = q
        _ctx_history_append(ctx2, "assistant", q)
        ctx2["last_reply"] = q
        ctx2["last_question_key"] = "AFTER_FLIGHT_NEXT_STEP"
        return _persist_context(state, ctx2)

    # ✅ Normal path: use LLM planner
    plan = plan_next_step(
        user_input=user_text,
        state={
            "slots": slots,
            "results": results,
            "history": history,
            "last_reply": last_reply,
            "last_question": last_q,
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
        _ctx_history_append(ctx2, "assistant", q)
        ctx2["last_reply"] = q
        ctx2["last_question_key"] = "ASK_USER"
        return _persist_context(state, ctx2)

    return _persist_context(state, ctx2)


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

    from app.agents.flights import run_flights_agent

    try:
        data = run_flights_agent(flights_provider, s["origin"], s["destination"], s["date"])
    except FlightsUnknownLocationError as e:
        if getattr(e, "suggestions", None):
            opts = "\n".join(
                [
                    f"- {x.get('name')} ({x.get('iataCode')})"
                    + (f", {x.get('countryCode')}" if x.get("countryCode") else "")
                    for x in e.suggestions[:5]
                ]
            )
            q = (
                f"I couldn’t find a matching {e.field} for “{e.query}”. Did you mean one of these?\n"
                f"{opts}\n\nReply with the correct one."
            )
        else:
            q = (
                f"I couldn’t find a matching {e.field} for “{e.query}”. "
                "Please tell me a nearby major airport/city (e.g., ‘near Surat’, ‘near Pune’)."
            )
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "flights_unknown_location", {"field": e.field, "query": e.query})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

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

    # ✅ IMPORTANT: Fill country info from Amadeus destination
    # ✅ IMPORTANT: Fill country info from Amadeus destination
    _resolve_country_from_destination(state)

    # refresh slots after mutation
    s = state.get("slots", {}) or {}

    reply = (
        f"Here are flights from {s['origin']} to {s['destination']} on {s['date']} (cheapest first).\n"
        f"Cheapest: {cheapest.get('airline')} {cheapest.get('flight_no')} ₹{cheapest.get('price_inr')} "
        f"({cheapest.get('depart')} → {cheapest.get('arrive')}).\n\n"
        "What would you like next?\n"
        "1) Book a hotel\n"
        "2) See COVID updates for your destination country\n\n"
        "Reply with: hotel / covid / or no"
    )

    state["reply"] = reply
    state["next_question"] = "Hotel or COVID updates?"
    add_trace(state, "flights_ok", {"cheapest": cheapest, "slots": s})

    ctx2 = dict(ctx)
    _ctx_history_append(ctx2, "assistant", reply)
    ctx2["last_reply"] = reply
    ctx2["last_question_key"] = "AFTER_FLIGHT_NEXT_STEP"
    ctx2["slots"] = s

    return _persist_context(state, ctx2)



def node_hotels(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    required = ["city", "checkin", "checkout"]
    missing = [k for k in required if not s.get(k)]
    if missing:
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
    except HotelsUnknownLocationError as e:
        q = (
            f"I couldn’t find the city “{e.query}”. "
            "Please confirm the city and country/state (e.g., ‘Springfield, IL, USA’)."
        )
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "hotels_unknown_city", {"query": e.query})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    except Exception as e:
        q = f"Hotel search failed: {str(e)}. Try another city or dates."
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
    if per_night is not None:
        price_txt = f"₹{per_night}/night" + (f" (₹{total} total)" if total is not None else "")
    elif total is not None:
        price_txt = f"₹{total} total"
    else:
        price_txt = "Price unavailable"

    reply = (
        f"Here are hotels in {s['city']} from {s['checkin']} to {s['checkout']} (cheapest first).\n"
        f"Cheapest: {name} {price_txt}{rating_txt}.\n\n"
        "Do you want to arrange transport from airport to hotel?"
    )
    state["reply"] = reply
    state["next_question"] = "Do you want to arrange transport from airport to hotel?"
    add_trace(state, "hotels_ok", {"cheapest": cheapest})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    ctx["last_question_key"] = "ASK_CABS_AFTER_HOTELS"
    return _persist_context(state, ctx)


def node_cabs(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    required = ["pickup", "dropoff"]
    missing = [k for k in required if not s.get(k)]
    if missing:
        if "dropoff" in missing and s.get("pickup"):
            q = "Which hotel/area/address is your drop-off? (Example: 'Courtyard Mumbai Airport' or 'BOM' or full address)"
        elif "pickup" in missing:
            q = "Where should the cab pick you up from? (Example: 'Mumbai Airport' or 'BOM')"
        else:
            q = f"I need: {', '.join(missing)} to search transfers."
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_missing", {"missing": missing})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    try:
        data = run_cabs_agent(cabs_provider, s["pickup"], s["dropoff"])
    except CabsUnknownLocationError as e:
        if e.field == "pickup":
            q = (
                f"I couldn't resolve the pickup location “{e.query}”.\n\n"
                "Reply with an airport/city (e.g., 'Mumbai Airport' / 'BOM') or a full pickup address.\n\n"
                "What should be your exact pickup?"
            )
        else:
            q = (
                f"I couldn't resolve the drop-off location “{e.query}”.\n\n"
                "Reply with a hotel/area/airport/city (e.g., 'Taj Lands End' / 'BOM') or a full drop-off address.\n\n"
                "What should be your exact drop-off?"
            )
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_unknown_location", {"field": e.field, "query": e.query})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    except Exception as e:
        q = f"Transfer search failed: {str(e)}"
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
        q = f"No transfers found for {s['pickup']} → {s['dropoff']}. Try changing pickup/dropoff?"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "cabs_none", {})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    reply = (
        f"Transfer options from {s['pickup']} to {s['dropoff']} (cheapest first).\n"
        f"Cheapest: {cheapest.get('vendor')} {cheapest.get('type')} ₹{cheapest.get('fare_inr')}.\n\n"
        "Anything else you want to add (flights/hotels/cabs/covid)?"
    )
    state["reply"] = reply
    state["next_question"] = "Anything else you want to add?"
    add_trace(state, "cabs_ok", {"cheapest": cheapest})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    ctx["last_question_key"] = "OPEN_ENDED"
    return _persist_context(state, ctx)


def node_covid(state: TravelState) -> TravelState:
    s = state.get("slots", {}) or {}
    ctx = state.get("updated_context") or state.get("convo_context") or {}

    # ✅ Prefer country filled from flights (Amadeus-derived)
    country = s.get("country") or s.get("destination_country")

    # If still missing, try to fill again from destination (Amadeus)
    if not country:
        _resolve_country_from_destination(state)
        s = state.get("slots", {}) or {}
        country = s.get("country") or s.get("destination_country")

    if not country:
        q = "I couldn’t infer the destination country from your flight destination. Please tell me the country name for COVID updates."
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "covid_missing_country", {"slots": s})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        ctx["last_question_key"] = "ASK_COVID_COUNTRY"
        return _persist_context(state, ctx)

    try:
        data = run_covid_agent(covid_provider, country)
    except Exception as e:
        q = f"Sorry — I couldn’t fetch COVID updates for {country}. ({str(e)})"
        state["reply"] = q
        state["next_question"] = q
        add_trace(state, "covid_error", {"country": country, "error": str(e)})
        _ctx_history_append(ctx, "assistant", q)
        ctx["last_reply"] = q
        return _persist_context(state, ctx)

    state.setdefault("results", {})
    state["results"]["covid"] = data

    totals = data.get("totals") or {}
    reply = (
        f"COVID update for {data.get('location') or country}.\n"
        f"Total cases: {totals.get('cases')} | Total deaths: {totals.get('deaths')} | Active: {totals.get('active')}\n"
        f"Today: +{totals.get('todayCases')} cases, +{totals.get('todayDeaths')} deaths.\n\n"
        "Do you want to book a hotel as well? (Yes/No)"
    )

    state["reply"] = reply
    state["next_question"] = "Do you want to book a hotel as well? (Yes/No)"
    add_trace(state, "covid_ok", {"country": country})

    _ctx_history_append(ctx, "assistant", reply)
    ctx["last_reply"] = reply
    ctx["last_question_key"] = "ASK_HOTEL_AFTER_COVID"
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
    g.add_node("covid", node_covid)

    g.set_entry_point("master")

    g.add_conditional_edges("master", node_route, {
        "flights": "flights",
        "hotels": "hotels",
        "cabs": "cabs",
        "covid": "covid",
        "ask_user": END,
    })

    g.add_edge("flights", END)
    g.add_edge("hotels", END)
    g.add_edge("cabs", END)
    g.add_edge("covid", END)

    return g.compile()
