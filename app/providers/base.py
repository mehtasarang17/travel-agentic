from abc import ABC, abstractmethod

class FlightsProvider(ABC):
    @abstractmethod
    def search_flights(self, origin: str, destination: str, date_iso: str) -> list[dict]:
        ...

class HotelsProvider(ABC):
    @abstractmethod
    def search_hotels(self, city: str, checkin_iso: str, checkout_iso: str) -> list[dict]:
        ...

class CabsProvider(ABC):
    @abstractmethod
    def search_cabs(self, pickup: str, dropoff: str) -> list[dict]:
        ...
