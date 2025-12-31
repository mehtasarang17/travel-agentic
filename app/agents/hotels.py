from app.providers.base import HotelsProvider

def run_hotels_agent(provider: HotelsProvider, city: str, checkin_iso: str, checkout_iso: str) -> dict:
    hotels = provider.search_hotels(city, checkin_iso, checkout_iso)
    cheapest = hotels[0] if hotels else None
    return {"hotels": hotels, "cheapest": cheapest}
