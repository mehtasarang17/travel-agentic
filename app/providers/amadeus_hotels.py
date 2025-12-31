import os
from typing import Dict, List, Optional, Tuple

from amadeus import Client, ResponseError, Location
from app.providers.base import HotelsProvider

from app.providers.amadeus_flights import UnknownLocationError  # reuse same error type


class AmadeusHotelsProvider(HotelsProvider):
    """
    Zero hardcoding:
    - Resolve user city text -> IATA city code via Airport & City Search
    - Get hotelIds by cityCode
    - Fetch offers by hotelIds + dates
    :contentReference[oaicite:2]{index=2}
    """

    def __init__(self):
        self.client = Client(
            client_id=os.getenv("AMADEUS_CLIENT_ID"),
            client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
            hostname=os.getenv("AMADEUS_HOSTNAME", "test"),
        )
        self._city_cache: Dict[str, str] = {}

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _search_cities(self, keyword: str, max_items: int = 6) -> List[dict]:
        try:
            resp = self.client.reference_data.locations.get(
                keyword=keyword,
                subType=Location.CITY,  # only city codes
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
                "countryCode": address.get("countryCode"),
                "cityName": address.get("cityName"),
            })
        return out[:max_items]

    def _resolve_city_code(self, city_text: str) -> str:
        raw = (city_text or "").strip()
        if not raw:
            raise UnknownLocationError("city", city_text, [])

        # allow direct city IATA
        if len(raw) == 3 and raw.isalpha():
            return raw.upper()

        key = self._norm(raw)
        if key in self._city_cache:
            return self._city_cache[key]

        candidates = self._search_cities(raw)
        if not candidates and len(raw) >= 3:
            candidates = self._search_cities(raw[:3])

        if not candidates:
            raise UnknownLocationError("city", raw, [])

        code = candidates[0]["iataCode"].upper()
        self._city_cache[key] = code
        return code

    def _get_hotel_ids_by_city(self, city_code: str, limit: int = 15) -> List[str]:
        try:
            resp = self.client.reference_data.locations.hotels.by_city.get(cityCode=city_code)
        except ResponseError as e:
            raise RuntimeError(str(e))

        ids = [h.get("hotelId") for h in (resp.data or []) if h.get("hotelId")]
        return ids[:limit]

    def search_hotels(self, city: str, checkin: str, checkout: str, adults: int = 1) -> list[dict]:
        city_code = self._resolve_city_code(city)

        hotel_ids = self._get_hotel_ids_by_city(city_code, limit=15)
        if not hotel_ids:
            return []

        try:
            offers = self.client.shopping.hotel_offers_search.get(
                hotelIds=hotel_ids,
                adults=str(adults),
                checkInDate=checkin,
                checkOutDate=checkout,
            )
        except ResponseError as e:
            raise RuntimeError(str(e))

        nights = max(1, _nights_between(checkin, checkout))

        out = []
        for item in (offers.data or []):
            hotel_info = item.get("hotel", {}) or {}
            name = hotel_info.get("name") or "Unknown"
            rating = hotel_info.get("rating") or hotel_info.get("hotelRating")
            hotel_id = hotel_info.get("hotelId") or item.get("hotelId")

            cheapest_total = None
            cheapest_offer_id = None

            for off in (item.get("offers") or []):
                total = off.get("price", {}).get("total")
                if total is None:
                    continue
                try:
                    total_f = float(total)
                except Exception:
                    continue

                if cheapest_total is None or total_f < cheapest_total:
                    cheapest_total = total_f
                    cheapest_offer_id = off.get("id")

            if cheapest_total is None:
                continue

            out.append({
                "name": name,
                "city": city_code,
                "hotel_id": hotel_id,
                "rating": rating,
                "price_total_inr": cheapest_total,
                "price_per_night_inr": round(cheapest_total / nights, 2),
                "offer_id": cheapest_offer_id,
            })

        return sorted(out, key=lambda x: x["price_total_inr"])


def _nights_between(checkin_iso: str, checkout_iso: str) -> int:
    from datetime import date
    ci = date.fromisoformat(checkin_iso)
    co = date.fromisoformat(checkout_iso)
    return (co - ci).days
