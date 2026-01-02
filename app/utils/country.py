import pycountry
from typing import Optional


def iso2_to_country_name(code: str) -> Optional[str]:
    """
    Convert ISO-2 country code (e.g. 'IN') to country name ('India')
    """
    if not code:
        return None

    try:
        country = pycountry.countries.get(alpha_2=code.upper())
        return country.name if country else None
    except Exception:
        return None
