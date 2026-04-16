from schemas.product_schema import ProductRecordV0, validate_product_record


def test_validate_product_record_passes_for_valid_record():
    record = ProductRecordV0(
        doc_id="doc-1",
        product_id=["P1"],
        document_type="Product Documents",
        title="Sample",
        attributes={"voltage": "24V"},
        source_trace=[],
        quality_flags=[],
    ).to_dict()
    ok, errors = validate_product_record(record)
    assert ok is True
    assert errors == []


def test_validate_product_record_fails_missing_required():
    record = {
        "doc_id": "doc-1",
        "product_id": ["P1"],
        "document_type": "Product Documents",
    }
    ok, errors = validate_product_record(record)
    assert ok is False
    assert any("missing_fields" in err for err in errors)
