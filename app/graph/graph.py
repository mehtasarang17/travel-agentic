from langgraph.graph import StateGraph, END
from app.graph.state import TravelState
from app.graph.intent import (
    detect_intent_and_slots,
    normalize_yes_no,
    extract_checkout_if_only_date,
    norm_city,
)

# ✅ Providers (Flights = Amadeus realtime, Hotels/Cabs = mock for now)
from app.providers.amadeus_flights import AmadeusFlightsProvider
from app.providers.amadeus_hotels import AmadeusHotelsProvider
from app.providers.mock_cabs import MockCabsProvider

# ✅ Agents (make sure these imports exist to avoid NameError)
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


def node_master_parse(state: TravelState) -> TravelState:
    """
    Master parses user input.
    - If user says Yes/No, use convo_context.last_question_key to route.
    - If we are awaiting a specific slot (checkout/origin/hotel selection), fill it.
    - Else do standard intent+slot detection.
    """
    user_text = state["user_input"]
    ctx = state.get("convo_context", {}) or {}

    yn = normalize_yes_no(user_text)

    # 1) Slot-fill: waiting for hotel checkout date
    if ctx.get("awaiting") == "HOTEL_CHECKOUT":
        checkout = extract_checkout_if_only_date(user_text)
        if checkout:
            partial = ctx.get("partial_slots") or {}
            state["intent"] = "hotels"
            state["slots"] = {**partial, "checkout": checkout}
            add_trace(state, "master_slotfill_checkout", {"checkout": checkout})

            state["updated_context"] = {
                **ctx,
                "awaiting": None,
                "partial_slots": None,
                "last_question_key": None,
            }
            return state

    # 2) Slot-fill: waiting for flight origin/current location
    if ctx.get("awaiting") == "FLIGHT_ORIGIN":
        origin = norm_city(user_text.strip())
        if origin:
            partial = ctx.get("partial_slots") or {}
            state["intent"] = "flights"
            state["slots"] = {**partial, "origin": origin}
            add_trace(state, "master_slotfill_origin", {"origin": origin})

            state["updated_context"] = {
                **ctx,
                "awaiting": None,
                "partial_slots": None,
                "last_question_key": None,
            }
            return state

    # ✅ 2.5) Slot-fill: waiting for user to select hotel for cab booking
    # User types: "Courtyard by Marriott Mumbai International Airport"
    if ctx.get("awaiting") == "CAB_HOTEL_SELECTION":
        chosen_hotel = user_text.strip()
        partial = ctx.get("partial_slots") or {}
        cab_city = partial.get("cab_city") or "Destination"
        candidates = partial.get("hotel_candidates") or []

        # Route to cabs with generated pickup/dropoff
        state["intent"] = "cabs"
        state["slots"] = {
            "pickup": f"{cab_city} Airport",
            "dropoff": chosen_hotel
        }
        add_trace(state, "master_slotfill_cab_hotel", {
            "pickup": f"{cab_city} Airport",
            "dropoff": chosen_hotel,
            "candidates_preview": candidates[:5],
        })

        state["updated_context"] = {
            **ctx,
            "awaiting": None,
            "partial_slots": None,
            "last_question_key": None,
            "pending": None,
        }
        return state

    # 3) Yes/No routing based on last question
    if yn is not None and ctx.get("last_question_key"):
        key = ctx["last_question_key"]
        pending = ctx.get("pending") or {}

        if yn == "no":
            state["intent"] = "unknown"
            state["slots"] = {}
            state["reply"] = "Got it — no problem. What would you like to do next (flights / hotels / cabs)?"
            state["next_question"] = "What would you like to do next?"
            add_trace(state, "master_yesno_no", {"key": key})

            state["updated_context"] = {
                **ctx,
                "last_question_key": None,
                "pending": None,
                "awaiting": None,
                "partial_slots": None,
            }
            return state

        # yn == "yes"
        if key == "ASK_HOTEL_AFTER_FLIGHT":
            state["intent"] = "hotels"
            state["slots"] = pending.get("hotels_slots", {})
            add_trace(state, "master_yesno_yes", {"route": "hotels", "slots": state["slots"]})
            return state

        if key == "ASK_FLIGHT_AFTER_HOTEL":
            state["intent"] = "flights"
            state["slots"] = pending.get("flights_slots", {})
            add_trace(state, "master_yesno_yes", {"route": "flights", "slots": state["slots"]})
            return state

        if key == "ASK_CAB_AFTER_HOTEL":
            state["intent"] = "cabs"
            state["slots"] = pending.get("cabs_slots", {})
            add_trace(state, "master_yesno_yes", {"route": "cabs", "slots": state["slots"]})
            return state

        if key == "ASK_CAB_AFTER_FLIGHT_AND_HOTEL":
            state["intent"] = "cabs"
            state["slots"] = pending.get("cabs_slots", {})
            add_trace(state, "master_yesno_yes", {"route": "cabs", "slots": state["slots"]})
            return state

        # ✅ NEW: After listing hotels, user says YES to cab → ask which hotel they selected
        if key == "ASK_CAB_AFTER_HOTELS_LIST":
            state["intent"] = "unknown"
            state["slots"] = {}
            state["reply"] = "Sure — which hotel have you selected? (Type the hotel name)"
            state["next_question"] = "Which hotel have you selected?"
            add_trace(state, "master_yesno_yes", {"route": "ask_hotel_selection"})

            state["updated_context"] = {
                **ctx,
                "last_question_key": None,
                "awaiting": "CAB_HOTEL_SELECTION",
                "partial_slots": {
                    "cab_city": pending.get("cab_city"),
                    "hotel_candidates": pending.get("hotel_candidates", []),
                },
                # keep pending optional; we don't need it anymore once awaiting is set
                "pending": None,
            }
            return state

    # 4) Default detection
    intent, slots = detect_intent_and_slots(user_text)
    state["intent"] = intent
    state["slots"] = slots
    add_trace(state, "master_parse", {"intent": intent, "slots": slots})
    return state

def node_route(state: TravelState) -> str:
    intent = state.get("intent", "unknown")
    if intent in {"flights", "hotels", "cabs"}:
        return intent
    return "unknown"


# ---------------------------
# Flights agent node
# ---------------------------
def node_flights(state: TravelState) -> TravelState:
    s = state.get("slots", {})
    ctx = state.get("convo_context", {}) or {}

    missing = [k for k in ["origin", "destination", "date"] if not s.get(k)]
    if missing:
        state["reply"] = f"I can help with flights, but I’m missing: {', '.join(missing)}."
        state["next_question"] = "Please provide those details (e.g., 'Delhi to Mumbai on 2026-01-15')."
        add_trace(state, "flights_missing", {"missing": missing})
        return state

    # ✅ IMPORTANT: local import prevents NameError forever
    from app.agents.flights import run_flights_agent

    data = run_flights_agent(flights_provider, s["origin"], s["destination"], s["date"])
    state.setdefault("results", {})
    state["results"]["flights"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        state["reply"] = f"No flights found for {s['origin']} → {s['destination']} on {s['date']}."
        state["next_question"] = "Do you want to try different dates?"
        add_trace(state, "flights_none", {})
        return state

    state["reply"] = (
        f"Here are flights from {s['origin']} to {s['destination']} on {s['date']} (cheapest first).\n"
        f"Cheapest: {cheapest['airline']} {cheapest['flight_no']} ₹{cheapest['price_inr']} "
        f"({cheapest['depart']} → {cheapest['arrive']}).\n\n"
        "Do you want to book a hotel as well? (Yes/No)"
    )
    state["next_question"] = "Do you want to book a hotel as well?"

    state["updated_context"] = {
        **ctx,
        "last_question_key": "ASK_HOTEL_AFTER_FLIGHT",
        "awaiting": None,
        "partial_slots": None,
        "pending": {
            "hotels_slots": {
                "city": s["destination"],
                "checkin": s["date"],
            }
        },
    }
    add_trace(state, "flights_agent", {"cheapest": cheapest})
    return state



# ---------------------------
# Hotels agent node
# ---------------------------
def node_hotels(state: TravelState) -> TravelState:
    s = state.get("slots", {})
    ctx = state.get("convo_context", {}) or {}

    if not s.get("city"):
        state["reply"] = "Sure — which city do you want to book the hotel in?"
        state["next_question"] = "Please tell me the city."
        add_trace(state, "hotels_missing", {"missing": ["city"]})
        return state

    # If user didn't give both dates at all
    if not s.get("checkin") and not s.get("checkout"):
        state["reply"] = "Please provide your check-in and check-out dates (e.g., 'from Jan 15 2026 to Jan 20 2026')."
        state["next_question"] = "Please enter your check-in and check-out dates."
        add_trace(state, "hotels_missing", {"missing": ["checkin", "checkout"]})
        return state

    # If user gave checkin but forgot checkout => store memory + ask
    if s.get("checkin") and not s.get("checkout"):
        state["reply"] = "Please enter last day of your stay (check-out date)."
        state["next_question"] = "What is your check-out date?"
        add_trace(state, "hotels_need_checkout", {"city": s["city"], "checkin": s["checkin"]})

        state["updated_context"] = {
            **ctx,
            "awaiting": "HOTEL_CHECKOUT",
            "partial_slots": {"city": s["city"], "checkin": s["checkin"]},
            "last_question_key": None,
        }
        return state

    # ✅ Call hotels agent (Amadeus provider underneath)
    try:
        data = run_hotels_agent(hotels_provider, s["city"], s["checkin"], s["checkout"])
    except Exception as e:
        state["reply"] = (
            f"Hotel search failed for {s['city']} ({s['checkin']} → {s['checkout']}).\n"
            f"Error: {str(e)}\n\n"
            "Try a different city (IATA code like DEL/BOM) or different dates."
        )
        state["next_question"] = "Do you want to try a different city or dates?"
        add_trace(state, "hotels_error", {"error": str(e)})
        return state

    state.setdefault("results", {})
    state["results"]["hotels"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        state["reply"] = f"No hotels found in {s['city']} for {s['checkin']} → {s['checkout']}."
        state["next_question"] = "Do you want to try different dates or a different city?"
        add_trace(state, "hotels_none", {})
        return state

    # ✅ Safe field reads (Amadeus may omit rating)
    name = cheapest.get("name", "Unknown Hotel")
    rating = cheapest.get("rating")
    per_night = cheapest.get("price_per_night_inr")
    total = cheapest.get("price_total_inr")

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

    # ✅ Collect hotel options to help with selection later (top 10)
    hotels_list = (data.get("hotels") or [])
    hotel_candidates = [h.get("name") for h in hotels_list if h.get("name")][:10]

    # ✅ Ask CAB question (instead of flight)
    state["reply"] = (
        f"Here are hotels in {s['city']} from {s['checkin']} to {s['checkout']} (cheapest first).\n"
        f"Cheapest: {name} {price_txt}{rating_txt}.\n\n"
        "Do you want to book a cab from your destination (airport) to your hotel? (Yes/No)"
    )
    state["next_question"] = "Do you want to book a cab from your destination (airport) to your hotel?"

    # ✅ Save memory for cab follow-up
    state["updated_context"] = {
        **ctx,
        "last_question_key": "ASK_CAB_AFTER_HOTELS_LIST",
        "awaiting": None,
        "partial_slots": None,
        "pending": {
            "cab_city": s["city"],                 # used later to build pickup = "<city> Airport"
            "hotel_candidates": hotel_candidates,  # optional validation/suggestions
        },
    }

    add_trace(state, "hotels_agent", {"cheapest": cheapest, "candidates": hotel_candidates})
    return state



# ---------------------------
# Cabs agent node
# ---------------------------
def node_cabs(state: TravelState) -> TravelState:
    s = state.get("slots", {})
    ctx = state.get("convo_context", {}) or {}

    missing = [k for k in ["pickup", "dropoff"] if not s.get(k)]
    if missing:
        state["reply"] = (
            f"I can help with cabs, but I’m missing: {', '.join(missing)}.\n"
            "Example: 'cab from Delhi airport to Taj Delhi'"
        )
        state["next_question"] = "Please provide pickup and dropoff."
        add_trace(state, "cabs_missing", {"missing": missing})
        return state

    data = run_cabs_agent(cabs_provider, s["pickup"], s["dropoff"])
    state.setdefault("results", {})
    state["results"]["cabs"] = data

    cheapest = data.get("cheapest")
    if not cheapest:
        state["reply"] = f"No cab options found for {s['pickup']} → {s['dropoff']}."
        state["next_question"] = "Do you want to try a different pickup/dropoff?"
        add_trace(state, "cabs_none", {})
        return state

    state["reply"] = (
        f"Cab options from {s['pickup']} to {s['dropoff']} (cheapest first).\n"
        f"Cheapest: {cheapest['vendor']} {cheapest['type']} ₹{cheapest['fare_inr']} (ETA {cheapest['eta_min']} min).\n\n"
        "Anything else you want to add (hotel/flight)?"
    )
    state["next_question"] = "Anything else you want to add?"

    state["updated_context"] = {
        **ctx,
        "last_question_key": None,
        "awaiting": None,
        "partial_slots": None,
        "pending": None,
    }

    add_trace(state, "cabs_agent", {"cheapest": cheapest})
    return state


def node_unknown(state: TravelState) -> TravelState:
    ctx = state.get("convo_context", {}) or {}
    state["reply"] = (
        "I can help with:\n"
        "1) Flights (e.g., 'Delhi to Mumbai on Jan 15 2026')\n"
        "2) Hotels (e.g., 'hotel in Delhi from Jan 15 2026 to Jan 20 2026')\n"
        "3) Cabs (e.g., 'cab from Delhi airport to Taj Delhi')\n\n"
        "What would you like to do?"
    )
    state["next_question"] = "What would you like to do?"
    state["updated_context"] = {
        **ctx,
        "last_question_key": None,
        "awaiting": None,
        "partial_slots": None,
        "pending": None,
    }
    add_trace(state, "unknown", {})
    return state


# ---------------------------
# Build graph
# ---------------------------
def build_graph():
    g = StateGraph(TravelState)

    g.add_node("master_parse", node_master_parse)
    g.add_node("flights", node_flights)
    g.add_node("hotels", node_hotels)
    g.add_node("cabs", node_cabs)
    g.add_node("unknown", node_unknown)

    g.set_entry_point("master_parse")
    g.add_conditional_edges("master_parse", node_route, {
        "flights": "flights",
        "hotels": "hotels",
        "cabs": "cabs",
        "unknown": "unknown",
    })

    g.add_edge("flights", END)
    g.add_edge("hotels", END)
    g.add_edge("cabs", END)
    g.add_edge("unknown", END)

    return g.compile()
