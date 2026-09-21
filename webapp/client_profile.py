"""Client demographics: country, region, group and acquisition channel.

WHY REGION RATHER THAN SERVER

Every breakdown in this product has been by server, which is an operational
artefact -- it says which machine an account happens to live on, not who the
client is. A desk asking "where is our risk" means geography, and a compliance
question about concentration is almost always regional. Region is the business
dimension; server is plumbing.

THE COUNTRY FIELD IS DIRTY AND MUST BE NORMALISED

The raw values are inconsistent in ways that silently split a country in two:

    CN (116,327)   China (799)
    MY (4,898)     Malaysia (475)
    eg, hk         lowercase ISO codes

Aggregating those as-is understates China by 799 accounts and scatters smaller
markets across several spellings. Everything here resolves to an ISO-3166 alpha-2
code first, then to a region, so a total is a total.
"""

from __future__ import annotations

import time

import pandas as pd

#: Full names and common variants seen in the data, mapped to ISO alpha-2.
#: Extended as new spellings appear rather than guessed at by fuzzy matching --
#: a wrong country is worse than an unknown one.
COUNTRY_ALIASES: dict[str, str] = {
    "CHINA": "CN", "PEOPLES REPUBLIC OF CHINA": "CN", "P.R. CHINA": "CN",
    "MALAYSIA": "MY", "INDONESIA": "ID", "THAILAND": "TH", "VIETNAM": "VN",
    "VIET NAM": "VN", "TAIWAN": "TW", "HONG KONG": "HK", "SINGAPORE": "SG",
    "INDIA": "IN", "TURKEY": "TR", "EGYPT": "EG", "RUSSIA": "RU",
    "RUSSIAN FEDERATION": "RU", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB",
    "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "USA": "US",
    "SOUTH KOREA": "KR", "KOREA": "KR", "JAPAN": "JP", "PHILIPPINES": "PH",
    "PAKISTAN": "PK", "BANGLADESH": "BD", "NIGERIA": "NG", "SOUTH AFRICA": "ZA",
    "UNITED ARAB EMIRATES": "AE", "UAE": "AE", "SAUDI ARABIA": "SA",
    "AUSTRALIA": "AU", "NEW ZEALAND": "NZ", "CANADA": "CA", "BRAZIL": "BR",
    "MEXICO": "MX", "ARGENTINA": "AR", "COLOMBIA": "CO", "CHILE": "CL",
    "GERMANY": "DE", "FRANCE": "FR", "SPAIN": "ES", "ITALY": "IT",
    "POLAND": "PL", "UKRAINE": "UA", "KAZAKHSTAN": "KZ", "SRI LANKA": "LK",
    "NEPAL": "NP", "CAMBODIA": "KH", "MYANMAR": "MM", "LAOS": "LA",
}

#: Region per ISO code. Coarse on purpose -- five buckets a desk actually uses,
#: not twenty-two UN subregions nobody reports on.
REGION_OF_COUNTRY: dict[str, str] = {
    **{code: "APAC" for code in (
        "CN", "HK", "TW", "JP", "KR", "SG", "MY", "ID", "TH", "VN", "PH", "IN",
        "PK", "BD", "LK", "NP", "KH", "MM", "LA", "AU", "NZ", "MN", "BN", "MO")},
    **{code: "EMEA" for code in (
        "GB", "DE", "FR", "ES", "IT", "NL", "BE", "CH", "AT", "SE", "NO", "DK",
        "FI", "PL", "CZ", "HU", "RO", "GR", "PT", "IE", "RU", "UA", "TR", "KZ",
        "AE", "SA", "QA", "KW", "BH", "OM", "EG", "ZA", "NG", "KE", "MA", "IL",
        "JO", "LB", "CY", "BG", "HR", "RS", "SK", "SI", "LT", "LV", "EE")},
    **{code: "AMER" for code in (
        "US", "CA", "BR", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "UY", "PA",
        "CR", "DO", "GT", "BO", "PY")},
}

_PROFILE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_PROFILE_TTL = 1800.0


def normalise_country(value) -> str:
    """Any spelling to an ISO alpha-2 code, or 'UNKNOWN'."""
    text = str(value or "").strip().upper()
    if not text:
        return "UNKNOWN"
    if len(text) == 2 and text.isalpha():
        return text
    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]
    if len(text) == 3 and text.isalpha():
        return {"CHN": "CN", "USA": "US", "GBR": "GB", "IDN": "ID", "THA": "TH",
                "VNM": "VN", "MYS": "MY", "IND": "IN", "RUS": "RU"}.get(text, "UNKNOWN")
    return "UNKNOWN"


def region_of(country_code: str) -> str:
    return REGION_OF_COUNTRY.get(str(country_code or "").upper(), "OTHER")


def load_profiles(database: str) -> pd.DataFrame:
    """Account demographics for one server, cached.

    MT4 and MT5 name these columns slightly differently; both are mapped onto
    one shape so downstream code never branches on platform.
    """
    cached = _PROFILE_CACHE.get(database)
    if cached and time.time() - cached[0] < _PROFILE_TTL:
        return cached[1]

    from webapp.mysql_extract import _connection

    if database.startswith("mt5"):
        sql = ("SELECT login, `group`, country, city, state, lead_source, agent AS agent_account,"
               " registration AS registered, name FROM accounts")
    else:
        sql = ("SELECT login, `group`, country, city, state, lead_source, agent_account,"
               " regdate AS registered, name FROM accounts")

    connection = _connection(database, timeout=180)
    try:
        frame = pd.read_sql(sql, connection)
    except Exception:
        # Column sets differ slightly by version; fall back to the essentials.
        connection.close()
        connection = _connection(database, timeout=180)
        frame = pd.read_sql("SELECT login, `group`, country FROM accounts", connection)
        for column in ("city", "state", "lead_source", "agent_account", "registered", "name"):
            frame[column] = pd.NA
    finally:
        connection.close()

    frame["database"] = database
    frame["account_key"] = database + ":" + frame["login"].astype("int64").astype(str)
    frame["country_code"] = frame["country"].map(normalise_country)
    frame["region"] = frame["country_code"].map(region_of)
    _PROFILE_CACHE[database] = (time.time(), frame)
    return frame


def load_all_profiles(databases: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for database in databases:
        try:
            frames.append(load_profiles(database))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["account_key", "country_code", "region", "group"])
    return pd.concat(frames, ignore_index=True)


def attach_region(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Add country and region to any frame carrying `account_key`."""
    if frame.empty or profiles.empty or "account_key" not in frame.columns:
        result = frame.copy()
        result["region"] = "UNKNOWN"
        result["country_code"] = "UNKNOWN"
        return result
    lookup = profiles[["account_key", "country_code", "region", "group"]] \
        .drop_duplicates("account_key")
    merged = frame.merge(lookup, on="account_key", how="left")
    merged["region"] = merged["region"].fillna("UNKNOWN")
    merged["country_code"] = merged["country_code"].fillna("UNKNOWN")
    return merged
