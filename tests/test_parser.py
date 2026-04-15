# Regression tests for the advanced-rag parsing pipeline.
#
# Each test uses real text patterns observed in 100_sample_advanced_rag.csv to
# verify that the improved parser extracts sections, attributes, FAQ answers,
# and inline specs correctly.
#
# These tests do NOT depend on pandas, langdetect, or any external service and
# can be run with `pytest tests/test_parser.py` in isolation.

from preprocessing.parser import (
    _is_section_heading,
    _pre_normalize_text,
    parse_raw_content,
    process_row,
)


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

def test_detects_all_caps_heading():
    assert _is_section_heading("KEY FEATURES") == "KEY FEATURES"
    assert _is_section_heading("SPECIFICATIONS") == "SPECIFICATIONS"
    assert _is_section_heading("ORDERING INFORMATION") == "ORDERING INFORMATION"


def test_detects_title_case_heading():
    assert _is_section_heading("Power Supply") == "Power Supply"
    assert _is_section_heading("Product Overview") == "Product Overview"


def test_detects_numbered_heading():
    result = _is_section_heading("1. System Configuration")
    assert result is not None and "System Configuration" in result


def test_detects_colon_heading():
    assert _is_section_heading("Electrical:") == "Electrical"


def test_does_not_flag_prose_as_heading():
    # Sentence with lowercase words should not be a heading
    assert _is_section_heading("This device monitors lithium-ion battery off-gas events") is None


# ---------------------------------------------------------------------------
# Pre-normalization
# ---------------------------------------------------------------------------

def test_pre_normalize_removes_page_lines():
    text = "SPECIFICATIONS\nPage 1 of 2\nDimensions: 100mm"
    normalised = _pre_normalize_text(text)
    assert "Page 1 of 2" not in normalised
    assert "SPECIFICATIONS" in normalised


def test_pre_normalize_decodes_unicode_bullets():
    # \u2022 is the bullet character U+2022
    text = "\u2022 Early warning of lithium-ion battery failures"
    normalised = _pre_normalize_text(text)
    # Should have been converted to a dash-space prefix
    assert "Early warning" in normalised
    assert "\u2022" not in normalised


def test_pre_normalize_removes_copyright_line():
    text = "© 2023 Honeywell Inc. All rights reserved.\nKEY FEATURES"
    normalised = _pre_normalize_text(text)
    assert "Honeywell" not in normalised
    assert "KEY FEATURES" in normalised


# ---------------------------------------------------------------------------
# Key-value extraction
# ---------------------------------------------------------------------------

def test_extracts_simple_kv():
    text = "Voltage: 120V\nCurrent: 10A"
    parsed = parse_raw_content("d1", text)
    attrs = parsed["attributes_raw"]
    assert "voltage" in attrs
    assert "current" in attrs


def test_extracts_multiline_kv():
    """A value that continues on the next non-blank line should be merged."""
    text = "Notes: Suitable for outdoor use\nand corrosive environments\n\nOther: value"
    parsed = parse_raw_content("d2", text)
    assert "Suitable for outdoor use and corrosive environments" in str(
        parsed["attributes_raw"].get("notes", "")
    )


def test_kv_synonym_normalization():
    """Voltage synonyms should all map to 'voltage'."""
    for label in ("Volt", "Voltage", "Input Voltage", "Operating Voltage"):
        text = f"{label}: 24V"
        parsed = parse_raw_content("d3", text)
        assert "voltage" in parsed["attributes_raw"], f"Failed for label: {label!r}"


def test_kv_synonym_dimensions():
    for label in ("Dimension", "Dimensions", "Product Dimensions", "Size"):
        text = f"{label}: 100 x 50 x 30 mm"
        parsed = parse_raw_content("d4", text)
        assert "dimensions" in parsed["attributes_raw"], f"Failed for label: {label!r}"


# ---------------------------------------------------------------------------
# Bullet point extraction
# ---------------------------------------------------------------------------

def test_extracts_plain_bullets():
    text = "KEY FEATURES\n- Early warning\n- Calibration-free\n- Low power consumption"
    parsed = parse_raw_content("d5", text)
    bullets = parsed["attributes_raw"].get("bullet_items", [])
    assert "Early warning" in bullets or any("Early" in b for b in bullets)


def test_bullet_with_inline_kv_promoted():
    """Bullets of the form '- Voltage: 24V' should be promoted to kv, not bullet_items."""
    text = "SPECIFICATIONS\n- Voltage: 24V\n- Current: 2A"
    parsed = parse_raw_content("d6", text)
    assert "voltage" in parsed["attributes_raw"]
    assert "current" in parsed["attributes_raw"]


# ---------------------------------------------------------------------------
# FAQ / numbered Q&A extraction (pattern seen in CLSS Gateway FAQ doc)
# ---------------------------------------------------------------------------

def test_extracts_faq_questions():
    text = (
        "1. Which panels will the CLSS Gateway support?\n"
        "Universal CLSS Gateway will support ONYX AFP-3030 and AFP-2800 Panels.\n\n"
        "2. Will there be a price increase?\n"
        "No. There won't be any price increases.\n"
    )
    parsed = parse_raw_content("d7", text)
    questions = parsed["attributes_raw"].get("faq_questions", [])
    assert len(questions) >= 2
    assert any("panels" in q.lower() for q in questions)


# ---------------------------------------------------------------------------
# Fixed-width / space-aligned table parsing (Li-Ion Tamer spec table pattern)
# ---------------------------------------------------------------------------

def test_fixed_width_table_row_parsed():
    """Two-column fixed-width rows should be promoted to kv attributes."""
    text = (
        "CONTROLLER SPECIFICATIONS\n"
        "Dimensions [mm]  210 (W) x 113 (L) x 63 (H)\n"
        "Input power range  12 - 28 VDC\n"
        "Max sensors per controller  15\n"
    )
    parsed = parse_raw_content("d8", text)
    attrs = parsed["attributes_raw"]
    # At least one of the known rows should be in attributes
    found = any(k for k in attrs if k not in ("table_rows",))
    assert found, f"No kv attributes extracted from fixed-width table; got: {list(attrs.keys())}"


def test_section_grouping():
    """Lines between two headings should belong to their respective sections."""
    text = (
        "KEY FEATURES\n"
        "- Feature A\n"
        "- Feature B\n\n"
        "SPECIFICATIONS\n"
        "Weight: 1.5 kg\n"
    )
    parsed = parse_raw_content("d9", text)
    titles = [s["title"] for s in parsed["sections"]]
    assert "KEY FEATURES" in titles
    assert "SPECIFICATIONS" in titles

    spec_section = next(s for s in parsed["sections"] if s["title"] == "SPECIFICATIONS")
    assert "weight" in spec_section["attributes"]


# ---------------------------------------------------------------------------
# process_row fallback field handling
# ---------------------------------------------------------------------------

def test_process_row_fallback_fields():
    row = {
        "document_id": "abc-123",
        "description": "Voltage: 240V",
        "name": "Product X",
        "type": "spec",
    }
    parsed = process_row(row, row_index=7)
    assert parsed["doc_id"] == "abc-123"
    assert parsed["title"] == "Product X"
    assert parsed["document_type"] == "spec"
    assert "voltage" in parsed["attributes_raw"]


def test_process_row_uses_id_field():
    row = {
        "id": "dm_0-abc123",
        "title": "Gateway FAQ",
        "file_content": "SPECIFICATIONS\nBaud rate: 9600",
        "document_type": "Product Documents",
        "product_id": '["P001", "P002"]',
    }
    parsed = process_row(row, row_index=0)
    assert parsed["doc_id"] == "dm_0-abc123"
    assert parsed["product_id"] == ["P001", "P002"]
    # 'baud rate' should be extracted
    assert "baud rate" in parsed["attributes_raw"]


# ---------------------------------------------------------------------------
# Coverage regression: check that parsing the Li-Ion Tamer spec text yields
# useful attributes (this replicates the real document from the sample CSV).
# ---------------------------------------------------------------------------

LIION_TAMER_EXCERPT = """
LI-ION TAMER
RACK MONITOR
Lithium Ion Battery Rack Monitoring System

KEY FEATURES
- Early warning of lithium-ion battery failures
- Enable thermal runaway prevention with proper mitigation actions
- Single cell failure detection without electrical or mechanical contact of cells
- Low power consumption

CONTROLLER SPECIFICATIONS
Dimensions [mm]  210 (W) x 113 (L) x 63 (H)
Input power range  12 - 28 VDC
Max sensors per controller  15

POWER CONSUMPTION SPECIFICATIONS
Controller (no sensors)  2.4 W (@ 24VDC)
Sensor  275 mW (@ 5 VDC)
Fuse Rating  3.5 A

GAS DETECTION SPECIFICATIONS
Min. detection threshold  < 1 ppm/sec
Response time  5 seconds
""".strip()


def test_liion_tamer_section_count():
    parsed = parse_raw_content("liion-test", LIION_TAMER_EXCERPT)
    titles = [s["title"] for s in parsed["sections"]]
    assert "KEY FEATURES" in titles
    assert "CONTROLLER SPECIFICATIONS" in titles
    assert "GAS DETECTION SPECIFICATIONS" in titles


def test_liion_tamer_attribute_extraction():
    parsed = parse_raw_content("liion-test", LIION_TAMER_EXCERPT)
    attrs = parsed["attributes_raw"]
    # Bullets should be captured
    assert "bullet_items" in attrs or "faq_answers" in attrs or len(attrs) >= 2
    # At least some technical attributes from the spec block
    useful_keys = {k for k in attrs if k not in ("table_rows",)}
    assert len(useful_keys) >= 3, f"Too few useful attributes extracted: {useful_keys}"


def test_liion_tamer_bullet_count():
    parsed = parse_raw_content("liion-test", LIION_TAMER_EXCERPT)
    bullets = parsed["attributes_raw"].get("bullet_items", [])
    assert len(bullets) >= 3
