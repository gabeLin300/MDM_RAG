import csv
import json

from pim_export import export_pim_files, normalize_pim_records


def test_normalize_current_orchestrator_shape_filters_empty_values():
    extracted = {
        "P100": [
            {"voltage": "24V DC"},
            {"current": None},
            {"mounting_type": "DIN rail"},
        ],
        "P200": [{"weight": ""}],
    }

    records = normalize_pim_records(extracted)

    assert len(records) == 1
    assert records[0].product_id == "P100"
    assert [(attr.name, attr.value, attr.status) for attr in records[0].attributes] == [
        ("voltage", "24V DC", "APPROVED"),
        ("mounting_type", "DIN rail", "APPROVED"),
    ]


def test_normalize_approval_records_exports_only_approved_and_edited():
    extracted = [
        {
            "product_id": "P100",
            "attributes": [
                {"name": "voltage", "value": "24V", "status": "APPROVED", "reviewer": "sam"},
                {"name": "weight", "value": "250g", "status": "PENDING"},
                {"dimensions": {"value": "10 x 5 x 3 cm", "status": "EDITED", "notes": "fixed units"}},
                {"material": {"value": "plastic", "status": "REJECTED"}},
            ],
        }
    ]

    records = normalize_pim_records(extracted)

    assert len(records) == 1
    attrs = records[0].attributes
    assert [attr.name for attr in attrs] == ["voltage", "dimensions"]
    assert attrs[0].reviewer == "sam"
    assert attrs[1].notes == "fixed units"


def test_export_pim_files_writes_json_and_csv(tmp_path):
    extracted = {
        "P100": [
            {"voltage": "24V"},
            {"communication_protocols": "BACnet, Modbus"},
        ]
    }

    result = export_pim_files(extracted, tmp_path, basename="sample")

    assert result["record_count"] == 1
    assert result["attribute_count"] == 2

    payload = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "pim_export_v1"
    assert payload["records"][0]["product_id"] == "P100"
    assert payload["records"][0]["attributes"][0]["name"] == "voltage"

    with (tmp_path / "sample.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["product_id"] == "P100"
    assert rows[0]["attribute_name"] == "voltage"
    assert rows[0]["attribute_value"] == "24V"
