from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_launches_with_sample_data_and_sample_prediction():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert [m.label for m in at.metric][:2] == ["Students", "Mean attendance"]
    # first-launch sample prediction (80 conducted, 52 attended = 65% -> At Risk)
    assert any("Predicted: At Risk" in e.value for e in at.error)


def test_manual_input_validation_message():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.number_input[1].set_value(100).run()      # attended 100 > conducted 80
    assert any("cannot exceed" in e.value for e in at.error)


def test_threshold_change_reruns_cleanly():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.number_input[0].set_value(60.0).run()
    assert not at.exception
