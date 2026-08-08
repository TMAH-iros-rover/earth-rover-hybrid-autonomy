from earth_rover.utils.status import format_urban_status


def test_format_urban_status_handles_missing_values():
    text = format_urban_status({"mode": "NORMAL_DRIVE", "safe_linear": 0.2, "safe_angular": -0.1})

    assert text.startswith("[URBAN]")
    assert "mode=NORMAL_DRIVE" in text
    assert "target=None" in text
    assert "safe=(0.20,-0.10)" in text
