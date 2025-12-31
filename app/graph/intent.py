import re
from dateutil import parser as dtparser



CITY_ALIASES = {
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
}

def norm_city(x: str) -> str:
    if not x:
        return x
    k = x.strip().lower()
    return CITY_ALIASES.get(k, x.strip().title())

def parse_date_maybe(text: str) -> str | None:
    try:
        d = dtparser.parse(text, fuzzy=True, dayfirst=True)
        return d.date().isoformat()
    except Exception:
        return None

def detect_intent_and_slots(user_text: str) -> tuple[str, dict]:
    t = user_text.lower()

    # cab intent
    if "cab" in t or "taxi" in t:
        # crude extraction: "from X to Y"
        m = re.search(r"from\s+(.*?)\s+to\s+(.*)", t)
        slots = {}
        if m:
            slots["pickup"] = m.group(1).strip().title()
            slots["dropoff"] = m.group(2).strip().title()
        return "cabs", slots

    # hotel intent
    if "hotel" in t or "stay" in t:
        slots = {}
        # city: "in <city>"
        m = re.search(r"in\s+([a-zA-Z ]+?)(?:\s+from|\s+on|$)", t)
        if m:
            slots["city"] = norm_city(m.group(1))

        # dates: "from <date> to <date>"
        m2 = re.search(r"from\s+(.*?)\s+to\s+(.*)", t)
        if m2:
            slots["checkin"] = parse_date_maybe(m2.group(1))
            slots["checkout"] = parse_date_maybe(m2.group(2))
        else:
            # "on <date>" single date (insufficient for hotels)
            m3 = re.search(r"on\s+(.*)", t)
            if m3:
                slots["checkin"] = parse_date_maybe(m3.group(1))

        return "hotels", slots

    # flights intent
    if "flight" in t or "travel" in t or "from" in t:
        slots = {}
        m = re.search(r"from\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+?)(?:\s+on|\s+at|\s+for|$)", t)
        if m:
            slots["origin"] = norm_city(m.group(1))
            slots["destination"] = norm_city(m.group(2))

        # date: "on <date>" OR "... Jan 15th 2026"
        date = parse_date_maybe(user_text)
        if date:
            slots["date"] = date

        return "flights", slots

    return "unknown", {}

YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "haan", "ha"}
NO = {"no", "n", "nope", "nah", "na"}

def normalize_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    if t in YES:
        return "yes"
    if t in NO:
        return "no"
    return None

def parse_single_date(text: str) -> str | None:
    # if user only typed a date like "Jan 20 2026"
    try:
        d = dtparser.parse(text, fuzzy=False, dayfirst=True)
        return d.date().isoformat()
    except Exception:
        return None

def extract_checkout_if_only_date(user_text: str) -> str | None:
    # Accept "Jan 20th" as checkout response
    return parse_single_date(user_text)
