# app/agents/covid.py
from __future__ import annotations

from typing import Any, Dict

from app.providers.disease_covid import DiseaseCovidProvider, CovidProviderError


def run_covid_agent(provider: DiseaseCovidProvider, location: str) -> Dict[str, Any]:
    """
    For now we treat 'location' as a COUNTRY name for reliable results.
    If location is a city and disease.sh can't resolve it, caller should ask user for country.
    """
    bundle = provider.fetch_country_bundle(location)

    # Convert 'updated' epoch ms to readable string (optional)
    updated_ms = bundle.get("updated")
    updated_str = None
    try:
        if updated_ms:
            import datetime
            updated_str = datetime.datetime.utcfromtimestamp(int(updated_ms) / 1000).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    totals = bundle.get("totals") or {}
    last14 = bundle.get("last14_days") or []

    # Prepare a chart-friendly series
    series = [{"x": p["date"], "y": p["new_cases"]} for p in last14]

    return {
        "location": bundle.get("country") or location,
        "updated_utc": updated_str,
        "totals": totals,
        "last14_days": last14,
        "series_new_cases": series,
        "cheapest": None,  # keep schema consistent if your UI expects 'cheapest' sometimes
    }
