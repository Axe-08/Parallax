"""
Parallax Preprocessing & Text Normalizer
========================================
Non-destructive data widening and multilingual text canonicalization:
- Unicode NFKC normalization (handles French accents and Indic scripts)
- Top-level web domain stripping (.com, .in, .fr, .org, .co)
- Structural building/unit number extraction
- Token-sorted views for word-order invariance
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from parallax.preprocessing.transliteration import transliterate_brahmic_to_latin

_DOMAIN_PATTERN = re.compile(
    r"\.(com|in|org|co|net|io|fr|gov|edu|biz|info)\b",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(
    r"[\r\n\t,;:\"\'\[\]\(\)\{\}\*\#\-\_\/\\]",
)
_NUMBER_PATTERN = re.compile(
    r"\b[a-z]{0,2}[0-9]{1,6}[a-z]{0,2}\b",
    re.IGNORECASE,
)

_ORDINAL_MAP: dict[str, str] = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
}

_ABBREVIATION_MAP: dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "flr": "floor",
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "llc": "limited liability company",
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new hampshire",
    "nj": "new jersey",
    "nm": "new mexico",
    "ny": "new york",
    "nc": "north carolina",
    "nd": "north dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode island",
    "sc": "south carolina",
    "sd": "south dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west virginia",
    "wi": "wisconsin",
    "wy": "wyoming",
    # Indian States and Union Territories
    "mh": "maharashtra",
    "dl": "delhi",
    "ka": "karnataka",
    "gj": "gujarat",
    "wb": "west bengal",
    "up": "uttar pradesh",
    "mp": "madhya pradesh",
    "ap": "andhra pradesh",
    "ts": "telangana",
    "kl": "kerala",
    "rj": "rajasthan",
    "pb": "punjab",
    "hr": "haryana",
    "br": "bihar",
    "od": "odisha",
    "jh": "jharkhand",
    "as": "assam",
    "ch": "chandigarh",
    "uk": "uttarakhand",
    # Transliterated Indic state tokens
    "mhaaraashtr": "maharashtra",
    "maharashtr": "maharashtra",
    "krnaatk": "karnataka",
    "gujraat": "gujarat",
    "tmilnaadu": "tamil nadu",
    "dillii": "delhi",
}

_PREFIX_MARKER = re.compile(
    r"\b(?:h\.?no\.?|house\s+no\.?|plot\s+no\.?|plot|flat\s+no\.?|flat|shop\s+no\.?|s\.?no\.?|khasara\s+no\.?|no\.?|#)\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)",
    re.IGNORECASE,
)

_ORDINAL_CHECK = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
_POSTAL_INDIA = re.compile(r"\b([1-8][0-9]{5})\b")
_POSTAL_US = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")


def normalize_unicode(text: str | None) -> str:
    """Standardize unicode representation using NFKC normalization."""
    if not text or pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text).strip())


def clean_soft_name(text: str | None) -> str:
    """
    Produce a relaxed soft representation of business name:
    - Lowercase + NFKC unicode
    - Strips domain extensions (e.g. burgersolution.com -> burgersolution)
    - Replaces brackets and punctuation with spaces
    - Collapses repeated whitespace
    """
    if not text or pd.isna(text):
        return ""

    raw = unicodedata.normalize("NFKC", str(text).strip().lower())
    raw = _DOMAIN_PATTERN.sub("", raw)
    raw = _PUNCTUATION_PATTERN.sub(" ", raw)
    return " ".join(raw.split())


def get_token_sorted_name(soft_name: str) -> str:
    """Return words sorted alphabetically to provide word-order transposition invariance."""
    tokens = soft_name.split()
    tokens.sort()
    return " ".join(tokens)


def extract_numbers(address: str | None) -> set[str]:
    """Extract numeric and alphanumeric building/flat numbers from an address string."""
    if not address or pd.isna(address):
        return set()
    raw = str(address).lower()
    matches = _NUMBER_PATTERN.findall(raw)
    return {m.strip() for m in matches if m.strip()}


def extract_primary_number(address: str | None) -> str | None:
    """Extract primary building, plot, flat, or municipal house identifier."""
    if not address or pd.isna(address):
        return None
    raw = str(address).strip()
    if not raw or raw.lower() == "nan":
        return None

    # 1. Prefix marker check (e.g. H.No. 8-2-283/4/2, Flat H-1, Plot No-21)
    m = _PREFIX_MARKER.search(raw)
    if m:
        val = m.group(1).lower().strip()
        val_clean = re.sub(r"[^a-z0-9]", "", val)
        if val_clean and not _ORDINAL_CHECK.match(val):
            return val_clean

    # 2. Check clauses for leading building/street number
    # (e.g. 28 Levant St, 4711 Howell Rd, 4A Gold Nest)
    clauses = [c.strip() for c in re.split(r"[,;]", raw) if c.strip()]
    for c in clauses:
        tokens = c.split()
        if not tokens:
            continue
        first = tokens[0].lower().lstrip("#")
        if _ORDINAL_CHECK.match(first):
            continue
        if re.search(r"\d", first):
            cleaned = re.sub(r"[^a-z0-9]", "", first)
            if cleaned:
                return cleaned
    return None


def extract_postal_code(address: str | None) -> str | None:
    """Extract 5-digit ZIP or 6-digit PIN code from address."""
    if not address or pd.isna(address):
        return None
    raw = str(address).strip()
    if not raw or raw.lower() == "nan":
        return None

    # Check India 6-digit PIN code
    in_matches = _POSTAL_INDIA.findall(raw)
    if in_matches:
        return str(in_matches[-1])

    # Check US 5-digit ZIP code
    us_matches = _POSTAL_US.findall(raw)
    if us_matches:
        return str(us_matches[-1])

    return None


def canonicalize_address(address: str | None) -> str:
    """Produce normalized lowercase address with ordinals and abbreviations folded."""
    if not address or pd.isna(address):
        return ""
    translit = transliterate_brahmic_to_latin(str(address))
    raw = unicodedata.normalize("NFKC", translit.strip().lower())
    raw = _PUNCTUATION_PATTERN.sub(" ", raw)
    tokens = raw.split()
    out: list[str] = []
    for t in tokens:
        t = _ORDINAL_MAP.get(t, t)
        t = _ABBREVIATION_MAP.get(t, t)
        out.append(t)
    return " ".join(out)


def extract_city_token(address: str | None) -> str | None:
    """Extract candidate city or municipality clause from an address string."""
    if not address or pd.isna(address):
        return None
    translit = transliterate_brahmic_to_latin(str(address))
    raw = str(translit).strip()
    if not raw or raw.lower() == "nan":
        return None
    clauses = [c.strip() for c in re.split(r"[,;\n]", raw) if c.strip()]
    if not clauses:
        return None
    candidate = clauses[-2] if len(clauses) >= 2 else clauses[-1]
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", candidate).strip().lower()
    words = [w for w in cleaned.split() if len(w) > 2]
    return " ".join(words) if words else None


def clean_address(address: str | None) -> str:
    """Produce a normalized lowercase address string with collapsed whitespace."""
    if not address or pd.isna(address):
        return ""
    raw = unicodedata.normalize("NFKC", str(address).strip().lower())
    raw = _PUNCTUATION_PATTERN.sub(" ", raw)
    return " ".join(raw.split())


def clean_transliterated_text(text: str | None) -> str:
    """Produce normalized Latin phonetic string from potentially multilingual text."""
    if not text or pd.isna(text):
        return ""
    translit = transliterate_brahmic_to_latin(str(text))
    return clean_soft_name(translit)


def widen_records_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived, parallel representation columns to a business records DataFrame
    without modifying the original raw columns.
    """
    enriched = df.copy()
    enriched["soft_name"] = enriched["business_name"].apply(clean_soft_name)
    enriched["token_sorted_name"] = enriched["soft_name"].apply(get_token_sorted_name)
    enriched["translit_name"] = enriched["business_name"].apply(clean_transliterated_text)
    enriched["clean_address"] = enriched["business_address"].apply(clean_address)
    enriched["translit_address"] = enriched["business_address"].apply(clean_transliterated_text)
    enriched["numbers"] = enriched["business_address"].apply(extract_numbers)
    enriched["primary_number"] = enriched["business_address"].apply(extract_primary_number)
    enriched["postal_code"] = enriched["business_address"].apply(extract_postal_code)
    enriched["canon_address"] = enriched["business_address"].apply(canonicalize_address)
    enriched["city_token"] = enriched["business_address"].apply(extract_city_token)
    enriched["is_addr_null"] = enriched["business_address"].isna().astype(int)
    return enriched
