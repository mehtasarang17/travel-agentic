import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from amadeus import Client, ResponseError, Location
from app.providers.base import CabsProvider


class UnknownLocationError(ValueError):
    def __init__(self, field: str, query: str, suggestions: List[dict]):
        super().__init__(f"Unknown {field}: {query}")
        self.field = field
        self.query = query
        self.suggestions = suggestions


class AmadeusCabsProvider(CabsProvider):
    """
    Amadeus Transfers (cab-like) pricing.
    - Resolves free text -> IATA code using Airport & City Search
    - Calls Transfers Search API to return priced offers

    Notes:
    - Works best for airport/city to airport/city routes
    - For hotel names without address, we may need user to provide an address/area
    """

    def __init__(self):
        self.client = Client(
            client_id=os.getenv("AMADEUS_CLIENT_ID"),
            client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
            hostname=os.getenv("AMADEUS_HOSTNAME", "test"),
        )
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _search_locations(self, keyword: str, max_items: int = 6) -> List[dict]:
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
            addr = it.get("address") or {}
            out.append({
                "name": it.get("name"),
                "iataCode": code,
                "subType": it.get("subType"),
                "countryCode": addr.get("countryCode"),
                "cityName": addr.get("cityName"),
            })

        # Prefer AIRPORT when user text includes "airport", else prefer CITY
        return out[:max_items]

    def _resolve_iata(self, text: str, field: str) -> str:
        raw = (text or "").strip()
        if not raw:
            raise UnknownLocationError(field, text, [])

        # direct IATA
        if len(raw) == 3 and raw.isalpha():
            return raw.upper()

        key = self._norm(raw)
        if key in self._cache:
            return self._cache[key]

        candidates = self._search_locations(raw)
        if not candidates and len(raw) >= 3:
            candidates = self._search_locations(raw[:3])

        if not candidates:
            raise UnknownLocationError(field, raw, [])

        wants_airport = "airport" in key
        # choose best candidate by subtype preference
        def score(item: dict) -> int:
            st = (item.get("subType") or "").upper()
            if wants_airport:
                return 2 if st == "AIRPORT" else 1 if st == "CITY" else 0
            return 2 if st == "CITY" else 1 if st == "AIRPORT" else 0

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        code = best["iataCode"].upper()
        self._cache[key] = code
        return code

    @staticmethod
    def _default_start_datetime_iso() -> str:
        """
        Transfers needs a future date-time. Use tomorrow 10:00 UTC.
        (You can improve later by passing user-selected time.)
        """
        now = datetime.now(timezone.utc)
        tmr = (now + timedelta(days=1)).date()
        dt = datetime(tmr.year, tmr.month, tmr.day, 10, 0, tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    def _parse_offer_price_inr(self, offer: dict) -> Optional[float]:
        """
        Amadeus transfers responses vary. Parse defensively.
        """
        # Common patterns: offer["quotation"]["monetaryAmount"] or offer["quotation"]["base"]["monetaryAmount"]
        q = offer.get("quotation") or {}
        for path in [
            ("monetaryAmount",),
            ("base", "monetaryAmount"),
            ("total", "monetaryAmount"),
        ]:
            cur = q
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                try:
                    return float(cur)
                except Exception:
                    pass
        return None

    def search_cabs(self, pickup: str, dropoff: str) -> list[dict]:
        """
        Returns list sorted cheapest first:
        {vendor,type,pickup,dropoff,eta_min,fare_inr}
        """
        start_code = self._resolve_iata(pickup, "pickup")
        end_code = self._resolve_iata(dropoff, "dropoff")

        body = {
            "startLocationCode": start_code,
            "endLocationCode": end_code,
            "transferType": "PRIVATE",   # you can also try "SHARED" depending on inventory
            "startDateTime": self._default_start_datetime_iso(),
            "passengers": 1,
            "currency": "INR",
        }

        try:
            # Amadeus SDK: transfers search is typically under shopping.transfer_offers.post
            resp = self.client.shopping.transfer_offers.post(body)
        except ResponseError as e:
            # Bubble as runtime error; your node can show a friendly message
            raise RuntimeError(str(e))

        offers = resp.data or []
        out = []

        for off in offers:
            price = self._parse_offer_price_inr(off)
            if price is None:
                continue

            vehicle = off.get("vehicle") or {}
            service = off.get("serviceProvider") or {}

            out.append({
                "vendor": service.get("name") or "Transfer",
                "type": vehicle.get("code") or vehicle.get("description") or "Car",
                "pickup": pickup,
                "dropoff": dropoff,
                "eta_min": None,
                "fare_inr": price,
            })

        out = sorted(out, key=lambda x: x["fare_inr"])
        return out
