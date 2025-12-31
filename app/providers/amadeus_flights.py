import os
from amadeus import Client, ResponseError
from app.providers.base import FlightsProvider

CITY_TO_IATA = {
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Hyderabad": "HYD",
    "Chennai": "MAA",
    "Kolkata": "CCU",
}

class AmadeusFlightsProvider(FlightsProvider):
    def __init__(self):
        self.client = Client(
            client_id=os.getenv("AMADEUS_CLIENT_ID"),
            client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
            hostname=os.getenv("AMADEUS_HOSTNAME", "test"),
        )

    def _to_iata(self, city_or_code: str) -> str | None:
        x = (city_or_code or "").strip()
        if len(x) == 3 and x.isalpha():
            return x.upper()
        return CITY_TO_IATA.get(x.title())

    def search_flights(self, origin: str, destination: str, date_iso: str) -> list[dict]:
        o = self._to_iata(origin)
        d = self._to_iata(destination)
        if not o or not d:
            raise ValueError("Unknown city. Please use airport codes like DEL, BOM, BLR.")

        try:
            # Flight Offers Search (GET)
            resp = self.client.shopping.flight_offers_search.get(
                originLocationCode=o,
                destinationLocationCode=d,
                departureDate=date_iso,
                adults=1,
                currencyCode="INR",
                max=15
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

            out.append({
                "airline": carrier,
                "flight_no": f"{carrier}{number}" if carrier and number else None,
                "origin": o,
                "destination": d,
                "date": date_iso,
                "depart": depart,
                "arrive": arrive,
                "price_inr": float(price) if price else None
            })

        # sort cheapest first
        out = [x for x in out if x["price_inr"] is not None]
        return sorted(out, key=lambda x: x["price_inr"])
