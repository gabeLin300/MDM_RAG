import re
from typing import Dict, List, Pattern

# pandas is an optional runtime dependency – the module must be importable without it
try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


HEADER_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE),
    re.compile(r"^\s*datasheet\s*$", re.IGNORECASE),
    # URL-only lines (e.g. "www.notifier.com.au 4 Trade Product list 2020")
    re.compile(r"^\s*www\.\S+\s*\d*\s*.*$", re.IGNORECASE),
    # Repeated dashes / underscores used as visual separators
    re.compile(r"^[-_=]{4,}\s*$"),
)

UNIT_NORMALIZATION: Dict[str, str] = {
    "inches": "in",
    "inch": "in",
    "millimeters": "mm",
    "millimeter": "mm",
    "centimeters": "cm",
    "centimeter": "cm",
    "volts": "V",
    "volt": "V",
    "amperes": "A",
    "ampere": "A",
    "amps": "A",
    "amp": "A",
    "watts": "W",
    "watt": "W",
    "hertz": "Hz",
    "kilograms": "kg",
    "kilogram": "kg",
    "pounds": "lbs",
    "pound": "lb",
}


def _coerce_text(text: object) -> str:
    """Safe text coercion – handles pandas NA/NaN without requiring pandas to be installed."""
    if text is None:
        return ""
    if _HAS_PANDAS and _pd.isna(text):  # type: ignore[union-attr]
        return ""
    return str(text)


def decode_unicode_escape(text: object) -> str:
    """Decode escaped sequences while tolerating malformed chunks."""
    value = _coerce_text(text)
    if not value:
        return ""
    decoded = value.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
    return decoded.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def remove_document_noise(text: str) -> str:
    """Drop common boilerplate lines such as page counters and repeated headers."""
    kept_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append("")
            continue
        if any(pattern.match(line) for pattern in HEADER_PATTERNS):
            continue
        kept_lines.append(raw_line)
    return "\n".join(kept_lines)


def fix_broken_words(text: str) -> str:
    """Repair words split by line breaks or dangling hyphens from PDF extraction."""
    fixed = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    fixed = re.sub(r"(\w)\n(\w)", r"\1 \2", fixed)
    return fixed


def normalize_units(text: str) -> str:
    """Normalize common unit spellings to canonical abbreviations."""
    normalized = text
    for source, target in UNIT_NORMALIZATION.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d+)\s+(in|mm|cm|V|A|W|Hz)\b", r"\1\2", normalized)
    return normalized


def clean_text(text: object) -> str:
    """Apply robust cleaning while preserving section-level structure."""
    value = decode_unicode_escape(text)
    if not value:
        return ""

    value = remove_document_noise(value)
    value = fix_broken_words(value)
    value = normalize_units(value)

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()