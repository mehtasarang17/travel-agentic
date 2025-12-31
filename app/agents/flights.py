from app.providers.base import FlightsProvider

def run_flights_agent(provider: FlightsProvider, origin: str, destination: str, date_iso: str) -> dict:
    flights = provider.search_flights(origin, destination, date_iso)
    cheapest = flights[0] if flights else None
    return {"flights": flights, "cheapest": cheapest}
