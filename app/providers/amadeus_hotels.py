import os
from amadeus import Client, ResponseError
from app.providers.base import HotelsProvider

# Minimal city -> IATA city code mapping (expand as needed)
CITY_TO_IATA = {
    "Delhi": "DEL",
    "New Delhi": "DEL",
    "Mumbai": "BOM",
    "Bombay": "BOM",
    "Bangalore": "BLR",
    "Bengaluru": "BLR",
    "Hyderabad": "HYD",
    "Chennai": "MAA",
    "Kolkata": "CCU",
    "Goa": "GOI",  # sometimes GOI used as airport; Amadeus hotel list expects cityCode; adjust if needed
}

class AmadeusHotelsProvider(HotelsProvider):
    def __init__(self):
        # Uses env vars if present (client_id/secret)
        self.client = Client(
            client_id=os.getenv("AMADEUS_CLIENT_ID"),
            client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
            hostname=os.getenv("AMADEUS_HOSTNAME", "test"),
        )

    def _to_city_code(self, city_or_code: str) -> str | None:
        x = (city_or_code or "").strip()
        if len(x) == 3 and x.isalpha():
            return x.upper()
        return CITY_TO_IATA.get(x.title())

    def search_hotels(self, city: str, checkin: str, checkout: str, adults: int = 1) -> list[dict]:
        """
        Returns list of hotels w/ prices (cheapest first).
        Flow:
          1) /reference-data/locations/hotels/by-city -> hotelIds
          2) /shopping/hotel-offers -> offers with price/rate
        """
        city_code = self._to_city_code(city)
        if not city_code:
            raise ValueError("Unknown city. Use IATA city code like DEL/BOM or supported city names.")

        try:
            # 1) Get hotels by city code
            hotels_by_city = self.client.reference_data.locations.hotels.by_city.get(
                cityCode=city_code
            )
            hotel_ids = [h.get("hotelId") for h in (hotels_by_city.data or []) if h.get("hotelId")]
            hotel_ids = hotel_ids[:15]  # keep it small for speed/rate limits

            if not hotel_ids:
                return []

            # 2) Get hotel offers (v3) for those hotelIds
            offers = self.client.shopping.hotel_offers_search.get(
                hotelIds=hotel_ids,
                adults=str(adults),
                checkInDate=checkin,
                checkOutDate=checkout,
            )
        except ResponseError as e:
            # Amadeus SDK provides nice error objects; stringify for now
            raise RuntimeError(str(e))

        out = []
        for hotel in (offers.data or []):
            hotel_info = hotel.get("hotel", {})
            name = hotel_info.get("name")
            rating = hotel_info.get("rating") or hotel_info.get("hotelRating")  # sometimes differs
            hotel_id = hotel_info.get("hotelId") or hotel.get("hotelId")

            # Pick cheapest offer for this hotel
            cheapest_price = None
            cheapest_offer = None
            for off in (hotel.get("offers") or []):
                price_total = off.get("price", {}).get("total")
                if price_total is None:
                    continue
                try:
                    p = float(price_total)
                except Exception:
                    continue
                if cheapest_price is None or p < cheapest_price:
                    cheapest_price = p
                    cheapest_offer = off

            if cheapest_price is None:
                continue

            # Convert total stay price to per-night (rough)
            nights = max(1, _nights_between(checkin, checkout))
            per_night = round(cheapest_price / nights, 2)

            out.append({
                "name": name,
                "city": city_code,
                "hotel_id": hotel_id,
                "rating": rating,
                "price_total_inr": cheapest_price,         # total for stay
                "price_per_night_inr": per_night,          # derived
                "offer_id": (cheapest_offer or {}).get("id"),
            })

        # Cheapest first
        return sorted(out, key=lambda x: x["price_total_inr"])


def _nights_between(checkin_iso: str, checkout_iso: str) -> int:
    from datetime import date
    ci = date.fromisoformat(checkin_iso)
    co = date.fromisoformat(checkout_iso)
    return (co - ci).days
