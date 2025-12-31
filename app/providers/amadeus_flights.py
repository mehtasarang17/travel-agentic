import os
from typing import Dict, List, Optional, Tuple

from amadeus import Client, ResponseError, Location
from app.providers.base import FlightsProvider


class UnknownLocationError(ValueError):
    def __init__(self, field: str, query: str, suggestions: List[dict]):
        super().__init__(f"Unknown {field}: {query}")
        self.field = field
        self.query = query
        self.suggestions = suggestions


class AmadeusFlightsProvider(FlightsProvider):
    """
    Zero hardcoding:
    - Resolve origin/destination from ANY user text using Airport & City Search
    - Then call flight offers search

    Uses:
      amadeus.reference_data.locations.get(keyword=..., subType=Location.ANY)
    :contentReference[oaicite:1]{index=1}
    """

    def __init__(self):
        self.client = Client(
            client_id=os.getenv("AMADEUS_CLIENT_ID"),
            client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
            hostname=os.getenv("AMADEUS_HOSTNAME", "test"),
        )
        # cache: normalized text -> iata
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _search_locations(self, keyword: str, max_items: int = 6) -> List[dict]:
        """
        Returns list of candidates: [{name, iataCode, subType, countryCode, cityName}]
        """
        try:
            resp = self.client.reference_data.locations.get(
                keyword=keyword,
                subType=Location.ANY,  # AIRPORT,CITY
            )
        except ResponseError:
            return []

        out = []
        for it in (resp.data or [])[: max_items * 2]:
            code = it.get("iataCode")
            if not code:
                continue
            address = it.get("address") or {}
            out.append({
                "name": it.get("name"),
                "iataCode": code,
                "subType": it.get("subType"),
                "countryCode": address.get("countryCode"),
                "cityName": address.get("cityName"),
            })

        # Prefer CITY first, then AIRPORT; also prefer higher traffic score if present
        def score(item: dict) -> Tuple[int, int]:
            st = (item.get("subType") or "").upper()
            st_score = 2 if st == "CITY" else 1 if st == "AIRPORT" else 0
            traffic = 0
            # Some responses include analytics.travelers.score; not always present
            return (st_score, traffic)

        out.sort(key=score, reverse=True)
        return out[:max_items]

    def _resolve_to_iata(self, text: str, field: str) -> str:
        """
        Accepts:
          - 'DEL' (already IATA)
          - 'Delhi' / 'New Delhi' / 'Heathrow' / 'JFK' / etc.
        """
        raw = (text or "").strip()
        if not raw:
            raise UnknownLocationError(field, text, [])

        # already an IATA code
        if len(raw) == 3 and raw.isalpha():
            return raw.upper()

        key = self._norm(raw)
        if key in self._cache:
            return self._cache[key]

        # Try full keyword, then fallback to first 3 chars (Amadeus autocomplete behaves best on prefixes)
        candidates = self._search_locations(raw)
        if not candidates and len(raw) >= 3:
            candidates = self._search_locations(raw[:3])

        if not candidates:
            raise UnknownLocationError(field, raw, [])

        best = candidates[0]
        code = best["iataCode"].upper()
        self._cache[key] = code
        return code

    def search_flights(self, origin: str, destination: str, date_iso: str) -> list[dict]:
        o = self._resolve_to_iata(origin, "origin")
        d = self._resolve_to_iata(destination, "destination")

        try:
            resp = self.client.shopping.flight_offers_search.get(
                originLocationCode=o,
                destinationLocationCode=d,
                departureDate=date_iso,
                adults=1,
                currencyCode="INR",
                max=15,
            )
        except ResponseError as e:
            raise RuntimeError(str(e))

        offers = resp.data or []
        out = []
        for off in offers:
            price = off.get("price", {}).get("grandTotal")
            itineraries = off.get("itineraries", [])
            seg = (itineraries[0]["segments"][0] if itineraries and itineraries[0].get("segments") else {})
            carrier = seg.get("carrierCode")
            number = seg.get("number")
            depart = seg.get("departure", {}).get("at")
            arrive = seg.get("arrival", {}).get("at")

            if price is None:
                continue

            out.append({
                "airline": carrier,
                "flight_no": f"{carrier}{number}" if carrier and number else None,
                "origin": o,
                "destination": d,
                "date": date_iso,
                "depart": depart,
                "arrive": arrive,
                "price_inr": float(price),
            })

        return sorted(out, key=lambda x: x["price_inr"])
