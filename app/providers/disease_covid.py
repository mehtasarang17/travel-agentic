from __future__ import annotations

import os
from typing import Any, Dict, Optional, List, Tuple
import requests
from datetime import datetime


class CovidProviderError(Exception):
    pass


class DiseaseCovidProvider:
    """
    Uses disease.sh API (no API key).
    Country-level stats are reliable.
    """

    BASE_URL = os.getenv("COVID_API_BASE", "https://disease.sh")

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.BASE_URL}{path}"
        r = requests.get(url, params=params or {}, timeout=self.timeout)
        if r.status_code >= 400:
            raise CovidProviderError(f"COVID API error {r.status_code}: {r.text[:200]}")
        return r.json()

    def get_latest_country(self, country: str) -> Dict[str, Any]:
        return self._get(f"/v3/covid-19/countries/{country}", params={"strict": "true"})

    def get_history_country_all(self, country: str) -> Dict[str, Any]:
        return self._get(f"/v3/covid-19/historical/{country}", params={"lastdays": "all"})

    @staticmethod
    def _parse_mdyy_to_iso(mdyy: str) -> str:
        """
        disease.sh date keys look like: '1/22/20'
        Convert to ISO: '2020-01-22'
        """
        dt = datetime.strptime(mdyy, "%m/%d/%y")
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _sorted_items_by_date(cumulative_by_date: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        Ensure we sort by actual date (not string order).
        Input keys like '1/2/20' will sort wrong if treated as strings.
        """
        items = []
        for k, v in (cumulative_by_date or {}).items():
            try:
                dt = datetime.strptime(k, "%m/%d/%y")
            except Exception:
                continue
            items.append((dt, int(v) if v is not None else 0))
        items.sort(key=lambda x: x[0])
        return [(d.strftime("%m/%d/%y"), v) for d, v in items]

    @classmethod
    def _compute_daily_new(cls, cumulative_by_date: Dict[str, int]) -> List[Dict[str, int]]:
        """
        Convert cumulative series -> list of {date, value} daily new series.
        Dates returned in ISO format (YYYY-MM-DD).
        """
        items = cls._sorted_items_by_date(cumulative_by_date)
        out: List[Dict[str, int]] = []
        prev = None
        for d, v in items:
            if prev is None:
                daily = 0
            else:
                daily = max(0, int(v) - int(prev))
            prev = v
            out.append({"date": cls._parse_mdyy_to_iso(d), "value": daily})
        return out

    def fetch_country_bundle(self, country: str) -> Dict[str, Any]:
        latest = self.get_latest_country(country)
        hist = self.get_history_country_all(country)

        timeline = (hist or {}).get("timeline") or {}
        cases_cum = timeline.get("cases") or {}
        deaths_cum = timeline.get("deaths") or {}

        # ✅ Full series (2020 → today)
        series_daily_cases = self._compute_daily_new(cases_cum)
        series_daily_deaths = self._compute_daily_new(deaths_cum)

        # last 14 days (optional)
        last14_cases = series_daily_cases[-14:] if len(series_daily_cases) >= 14 else series_daily_cases
        last14_deaths = series_daily_deaths[-14:] if len(series_daily_deaths) >= 14 else series_daily_deaths
        last14 = [
            {"date": last14_cases[i]["date"], "new_cases": last14_cases[i]["value"], "new_deaths": last14_deaths[i]["value"] if i < len(last14_deaths) else 0}
            for i in range(len(last14_cases))
        ]

        return {
            "country": latest.get("country") or country,
            "location": latest.get("country") or country,
            "updated": latest.get("updated"),  # epoch ms
            "totals": {
                "cases": latest.get("cases"),
                "deaths": latest.get("deaths"),
                "recovered": latest.get("recovered"),
                "active": latest.get("active"),
                "todayCases": latest.get("todayCases"),
                "todayDeaths": latest.get("todayDeaths"),
            },

            # ✅ This is what your HTML is expecting for full history
            "series_daily_cases": series_daily_cases,
            "series_daily_deaths": series_daily_deaths,

            # fallback / quick view
            "last14_days": last14,
        }
