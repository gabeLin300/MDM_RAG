import ast
import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Section-heading detection
# Handles four patterns found in this product-document corpus:
#   1. ALL CAPS headings (e.g. "SPECIFICATIONS", "KEY FEATURES")
#   2. Title Case headings without a colon (e.g. "Power Supply", "About Xtralis")
#   3. Numbered section headings  (e.g. "1. Introduction", "2.3 Wiring")
#   4. Original Title-or-mixed case with optional colon  (e.g. "Electrical:")
# ---------------------------------------------------------------------------
_ALL_CAPS_RE = re.compile(r"^(?P<title>[A-Z][A-Z0-9 /&()\-]{2,80})$")
_TITLE_CASE_RE = re.compile(r"^(?P<title>(?:[A-Z][a-z0-9]+(?:[ \-/][A-Z][a-z0-9]+){1,8})):?$")
_NUMBERED_RE = re.compile(r"^(?P<title>\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 /\-]{2,60}):?$")
_SECTION_RE = re.compile(r"^(?P<title>[A-Z][A-Za-z0-9 /&()\-]{2,60}):$")

# FAQ / numbered Q&A prompts  (e.g. "1. Which panels will…", "Q: What is…")
_FAQ_Q_RE = re.compile(r"^(?:\d+[.)]\s+|Q\s*[:.]?\s*)(?P<question>.{5,200})$", re.IGNORECASE)

# Key-value pairs – colon or dash separator
KV_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 /_()\-.]{1,80})\s*[:\-]\s*(?P<value>.+)$")

# Inline spec: "120V", "10A", "24VDC", "50/60Hz"  appearing after a label
_INLINE_SPEC_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 /_()\-.]{1,50})\s*[:\-]?\s*"
    r"(?P<value>\d+(?:[.,]\d+)?(?:\s*[-–/]\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:V(?:DC|AC)?|A|W|Hz|kg|lbs?|mm|cm|in|ft|°[CF]|%|ppm|mW|mA|kW|kHz|MHz|VA|°?C))"
    r"\b"
)

# Unicode bullet variants that survive escaping   (•, ▪, ◆, –)
BULLET_RE = re.compile(r"^\s*(?:[-*•▪◆–]|\u2022|\u25AA|\u25C6)\s+(?P<item>.+)$")

# Fixed-width / space-aligned table rows:
# at least 3 tokens separated by 2+ consecutive spaces
_FIXED_TABLE_RE = re.compile(r"^(.+?)(?:\s{2,})(.+?)(?:\s{2,})?(.*)$")

# ---------------------------------------------------------------------------
# Attribute synonym map – applied after lower-casing and de-noising the key
# ---------------------------------------------------------------------------
ATTRIBUTE_NAME_NORMALIZATION: Dict[str, str] = {
    # voltage
    "volt": "voltage", "voltage": "voltage", "input voltage": "voltage",
    "supply voltage": "voltage", "rated voltage": "voltage",
    "nominal voltage": "voltage", "operating voltage": "voltage",
    # current
    "amp": "current", "amps": "current", "ampere": "current",
    "current": "current", "rated current": "current", "input current": "current",
    "max current": "current",
    # power
    "watt": "power", "watts": "power", "power": "power",
    "power consumption": "power", "rated power": "power",
    "max power": "power", "input power": "power",
    # dimensions
    "dimension": "dimensions", "dimensions": "dimensions",
    "overall dimensions": "dimensions", "product dimensions": "dimensions",
    "size": "dimensions",
    # individual dimension axes
    "width": "width", "w": "width",
    "height": "height", "h": "height",
    "depth": "depth", "d": "depth",
    "length": "length", "l": "length",
    # weight
    "weight": "weight", "net weight": "weight", "gross weight": "weight",
    "product weight": "weight",
    # material
    "material": "material", "housing material": "material",
    "body material": "material",
    # capacity
    "capacity": "capacity", "storage capacity": "capacity",
    "battery capacity": "capacity",
    # temperature
    "temperature": "temperature", "operating temperature": "temperature",
    "storage temperature": "temperature", "ambient temperature": "temperature",
    "temp": "temperature",
    # frequency
    "frequency": "frequency", "hz": "frequency", "hertz": "frequency",
    # ip rating
    "ip rating": "ip_rating", "protection class": "ip_rating",
    "ingress protection": "ip_rating",
}


def parse_product_id(product_id: object) -> List[str]:
    """Parse product ids from JSON string, Python list string, scalar, or null."""
    if product_id is None:
        return []

    if isinstance(product_id, list):
        return [str(value).strip() for value in product_id if str(value).strip()]

    raw = str(product_id).strip()
    if not raw:
        return []

    raw = raw.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [str(value).strip() for value in parsed if str(value).strip()]
            if parsed is not None and str(parsed).strip():
                return [str(parsed).strip()]
        except Exception:
            continue

    return [raw]


def normalize_attribute_name(name: str) -> str:
    """Canonicalize noisy attribute key names to the synonym map."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    # strip trailing punctuation from keys like "Width (mm):" → "width (mm)"
    cleaned = cleaned.rstrip(":").strip()
    return ATTRIBUTE_NAME_NORMALIZATION.get(cleaned, cleaned)


def _is_section_heading(line: str) -> Optional[str]:
    """Return heading text if the line matches any supported heading pattern, else None."""
    stripped = line.strip()
    if not stripped:
        return None

    # Numbered section (highest priority – e.g. "1. Introduction")
    m = _NUMBERED_RE.match(stripped)
    if m:
        return m.group("title").strip()

    # All-caps heading (e.g. "KEY FEATURES", "SPECIFICATIONS")
    m = _ALL_CAPS_RE.match(stripped)
    if m:
        # Avoid treating short acronyms or data values as headings
        title = m.group("title").strip()
        if len(title) >= 4 and not re.search(r"\d", title):
            return title

    # Title Case multi-word (e.g. "Power Supply", "Product Overview")
    m = _TITLE_CASE_RE.match(stripped)
    if m:
        return m.group("title").strip()

    # General mixed heading with optional colon (original rule)
    m = _SECTION_RE.match(stripped)
    if m:
        return m.group("title").strip()

    return None


def _is_table_like_line(line: str) -> bool:
    """Detect pipe-delimited, tab-delimited, or fixed-width table rows."""
    stripped = line.strip()
    if stripped.count("|") >= 2:
        return True
    if "\t" in stripped and len([p for p in stripped.split("\t") if p.strip()]) >= 2:
        return True
    # Fixed-width: 3+ whitespace-separated tokens where gaps are 2+ spaces
    if re.search(r"\S\s{2,}\S", stripped):
        tokens = [t for t in re.split(r"\s{2,}", stripped) if t.strip()]
        if len(tokens) >= 2:
            return True
    return False


def _parse_table_line(line: str) -> List[str]:
    """Split a table-like line into its column tokens."""
    stripped = line.strip()
    if "|" in stripped:
        return [p.strip() for p in stripped.strip("|").split("|") if p.strip()]
    if "\t" in stripped:
        return [p.strip() for p in stripped.split("\t") if p.strip()]
    # Fixed-width split
    return [t.strip() for t in re.split(r"\s{2,}", stripped) if t.strip()]


def _table_row_to_kv(row: List[str]) -> Optional[Tuple[str, str]]:
    """
    For 2-column table rows that look like spec tables, promote them to
    key-value attributes.  Returns (canonical_key, value) or None.
    """
    if len(row) == 2:
        key = normalize_attribute_name(row[0])
        value = row[1].strip()
        if key and value:
            return key, value
    return None


def _extract_inline_specs(line: str) -> List[Tuple[str, str]]:
    """
    Extract technical specs written inline such as 'Input power range 12-28 VDC'
    or 'Baud rate 9600'.
    Returns a list of (canonical_key, value) pairs.
    """
    results = []
    for m in _INLINE_SPEC_RE.finditer(line):
        key = normalize_attribute_name(m.group("key"))
        value = m.group("value").strip()
        if key and value:
            results.append((key, value))
    return results


def _pre_normalize_text(raw: str) -> str:
    """
    Pre-parse normalization applied before line-by-line parsing.
    - Decodes escaped unicode  (\\n \\u00A9 etc.)
    - Normalises unicode bullet variants to ASCII hyphen-space
    - Collapses 3+ blank lines to 2
    - Strips obvious PDF artefacts (page counters, superscript footnote markers)
    Returns the cleaned text with structure preserved.
    """
    if not raw:
        return ""

    # Remove copyright / footer stamps BEFORE unicode decode (© is still intact here)
    text = re.sub(r"(?im)^[©®\u00a9\u00ae][^\n]{0,120}$", "", raw)
    text = re.sub(r"(?im)^\(c\)\s*\d{4}[^\n]{0,120}$", "", text)

    # Decode unicode escapes stored as literal backslash sequences  (\n, \u00A9…)
    try:
        text = text.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        pass

    # Normalise unicode bullets  (•  ▪  ◆  –  \u2022  \u25AA …) → "- "
    text = re.sub(r"[\u2022\u25AA\u25C6\u2023\u25E6]", "- ", text)

    # Strip page lines  ("Page 1 of 2", "www.example.com NNN")
    text = re.sub(r"(?im)^\s*page\s+\d+\s*(of\s+\d+)?\s*$", "", text)
    text = re.sub(r"(?im)^\s*www\.[^\s]+\s+\d+\s*$", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_non_empty(record: Dict[str, Any], candidates: Iterable[str], default: str = "") -> str:
    for field in candidates:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return default


def parse_raw_content(doc_id: str, text: object) -> Dict[str, Any]:
    """
    Parse text into sections, key-value attributes, and table-like rows.

    Parsing order (descending priority per line):
    1. Blank line          → section break / reset continuation
    2. Section heading     → start new section bucket (all-caps, title-case, numbered, original)
    3. FAQ question prompt → store under 'faq_questions' and start answer context
    4. Key-value pair      → add to attributes + optional section
    5. Bullet point        → add to bullet_items + optional section
    6. Table-like row      → store as table_rows; promote 2-col rows to kv attributes
    7. Continuation line   → append to previous kv value
    8. Prose line          → attach to current or implicit 'General' section
    """
    raw_text = "" if text is None else str(text)

    # Pre-normalise before splitting into lines
    normalised_text = _pre_normalize_text(raw_text)

    attributes: Dict[str, Any] = defaultdict(list)
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    current_attr_key: Optional[str] = None
    faq_answer_key: Optional[str] = None  # tracks numbered FAQ answer context

    lines = normalised_text.split("\n")
    for line in lines:
        stripped = line.strip()

        # ------------------------------------------------------------------ blank
        if not stripped:
            current_attr_key = None
            faq_answer_key = None
            continue

        # ------------------------------------------------------------------ section heading
        heading = _is_section_heading(stripped)
        if heading:
            current_section = {
                "title": heading,
                "content_lines": [],
                "attributes": {},
                "table_rows": [],
            }
            sections.append(current_section)
            current_attr_key = None
            faq_answer_key = None
            continue

        # ------------------------------------------------------------------ FAQ / numbered Q
        faq_match = _FAQ_Q_RE.match(stripped)
        if faq_match:
            question = faq_match.group("question").strip()
            attributes["faq_questions"].append(question)
            if current_section is not None:
                current_section["attributes"].setdefault("faq_questions", []).append(question)
                current_section["content_lines"].append(stripped)
            # Treat following lines as this question's answer until blank line
            faq_answer_key = "faq_answers"
            current_attr_key = None
            continue

        # ------------------------------------------------------------------ key-value pair
        kv_match = KV_RE.match(stripped)
        if kv_match:
            key = normalize_attribute_name(kv_match.group("key"))
            value = kv_match.group("value").strip()

            # Skip if the key looks like a very common English word that
            # is almost certainly a prose sentence, not a spec label.
            if len(key.split()) <= 6 and value:
                attributes[key].append(value)
                if current_section is not None:
                    current_section["attributes"].setdefault(key, []).append(value)
                    current_section["content_lines"].append(stripped)
                current_attr_key = key
                faq_answer_key = None
                continue

        # ------------------------------------------------------------------ bullet point
        bullet_match = BULLET_RE.match(stripped)
        if bullet_match:
            value = bullet_match.group("item").strip()
            # Try to extract a kv from bullet content  ("Voltage: 24V")
            inner_kv = KV_RE.match(value)
            if inner_kv:
                key = normalize_attribute_name(inner_kv.group("key"))
                val = inner_kv.group("value").strip()
                attributes[key].append(val)
                if current_section is not None:
                    current_section["attributes"].setdefault(key, []).append(val)
            else:
                attributes["bullet_items"].append(value)
                if current_section is not None:
                    current_section["attributes"].setdefault("bullet_items", []).append(value)
            if current_section is not None:
                current_section["content_lines"].append(stripped)
            current_attr_key = None
            continue

        # ------------------------------------------------------------------ table-like row
        if _is_table_like_line(stripped):
            row = _parse_table_line(stripped)
            attributes["table_rows"].append(row)
            if current_section is not None:
                current_section["table_rows"].append(row)
                current_section["content_lines"].append(stripped)

            # Promote 2-column rows to key-value attributes
            kv_from_table = _table_row_to_kv(row)
            if kv_from_table:
                key, value = kv_from_table
                attributes[key].append(value)
                if current_section is not None:
                    current_section["attributes"].setdefault(key, []).append(value)

            current_attr_key = None
            continue

        # ------------------------------------------------------------------ inline specs scan
        inline_specs = _extract_inline_specs(stripped)
        if inline_specs:
            for key, value in inline_specs:
                attributes[key].append(value)
                if current_section is not None:
                    current_section["attributes"].setdefault(key, []).append(value)
            # Still fall through to prose handling below so line is also stored

        # ------------------------------------------------------------------ FAQ answer continuation
        if faq_answer_key is not None:
            attributes[faq_answer_key].append(stripped)
            if current_section is not None:
                current_section["attributes"].setdefault(faq_answer_key, []).append(stripped)
                current_section["content_lines"].append(stripped)
            continue

        # ------------------------------------------------------------------ kv continuation
        if current_attr_key is not None:
            existing = attributes[current_attr_key]
            if existing:
                existing[-1] = f"{existing[-1]} {stripped}".strip()
            if current_section is not None:
                section_vals = current_section["attributes"].get(current_attr_key, [])
                if section_vals:
                    section_vals[-1] = f"{section_vals[-1]} {stripped}".strip()
                current_section["content_lines"].append(stripped)
            continue

        # ------------------------------------------------------------------ prose / unclassified
        if current_section is None:
            current_section = {
                "title": "General",
                "content_lines": [],
                "attributes": {},
                "table_rows": [],
            }
            sections.append(current_section)
        current_section["content_lines"].append(stripped)

    # Flatten multi-value lists: single entry → scalar, multiple → list
    flattened: Dict[str, Any] = {}
    for key, values in attributes.items():
        flattened[key] = values[0] if len(values) == 1 else values

    return {
        "doc_id": str(doc_id),
        "sections": sections,
        "attributes_raw": flattened,
        "text": raw_text,
    }


def process_row(row: Dict[str, Any], row_index: int = 0) -> Dict[str, Any]:
    """Build a parsed document dictionary from one CSV record."""
    doc_id = _first_non_empty(row, ("id", "doc_id", "document_id"), default=f"row-{row_index}")
    raw_text = _first_non_empty(
        row,
        ("file_content", "raw_text", "description", "specs", "text"),
        default="",
    )

    parsed = parse_raw_content(doc_id=doc_id, text=raw_text)
    parsed.update(
        {
            "title": _first_non_empty(row, ("title", "name")),
            "file_name": _first_non_empty(row, ("file_name", "source_file_name", "source_file")),
            "product_id": parse_product_id(row.get("product_id")),
            "document_type": _first_non_empty(row, ("document_type", "type")),
        }
    )
    return parsed