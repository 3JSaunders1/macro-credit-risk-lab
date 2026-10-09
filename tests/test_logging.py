"""Logging: run_main logs start, finish, and failures, and exits nonzero on error."""
import logging
import pytest
from utils.logging_utils import get_logger, run_main


def test_run_main_logs_start_and_finish(caplog):
    with caplog.at_level(logging.INFO):
        run_main(lambda: None)
    assert "Started" in caplog.text and "Finished in" in caplog.text


def test_run_main_exits_nonzero_and_logs_traceback(caplog):
    def boom():
        raise ValueError("bad input")
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
        run_main(boom)
    assert exc.value.code == 1
    assert "Failed after" in caplog.text and "bad input" in caplog.text


def test_logger_uses_short_module_name():
    assert get_logger("pipeline.scenario_engine").name == "scenario_engine"
