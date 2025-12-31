from .base import CabsProvider
import random

class MockCabsProvider(CabsProvider):
    def search_cabs(self, pickup: str, dropoff: str) -> list[dict]:
        types = ["Mini", "Sedan", "SUV"]
        vendors = ["Uber", "Ola", "BluSmart"]
        out = []
        for v in vendors:
            for t in types:
                out.append({
                    "vendor": v,
                    "type": t,
                    "pickup": pickup,
                    "dropoff": dropoff,
                    "eta_min": random.randint(3, 12),
                    "fare_inr": random.randint(180, 1200) + (0 if t == "Mini" else 200 if t == "Sedan" else 450)
                })
        return sorted(out, key=lambda x: x["fare_inr"])
