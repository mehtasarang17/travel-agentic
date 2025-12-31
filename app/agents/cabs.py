from app.providers.base import CabsProvider

def run_cabs_agent(provider: CabsProvider, pickup: str, dropoff: str) -> dict:
    cabs = provider.search_cabs(pickup, dropoff)
    cheapest = cabs[0] if cabs else None
    return {"cabs": cabs, "cheapest": cheapest}
