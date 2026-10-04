"""Dashboard smoke test: the app runs top to bottom without raising an exception."""
from streamlit.testing.v1 import AppTest


def test_dashboard_runs_without_errors():
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception
