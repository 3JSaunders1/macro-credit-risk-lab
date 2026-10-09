"""
Shared logging setup for every pipeline step.

- Messages go to stdout with a timestamp, level, and module name, so the
  Makefile's `tee` captures them in each run's log.
- The level is set by the LOG_LEVEL environment variable (default INFO).
- run_main() wraps a script's main(): it logs the start, the duration, and
  any failure (with traceback), and exits with code 1 on error.
"""
import logging
import os
import sys
import time
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(level: str | None = None) -> None:
    """Configure the root logger once."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level or os.getenv("LOG_LEVEL", "INFO"),
                        format=FORMAT, datefmt="%H:%M:%S", stream=sys.stdout)


def _script_name() -> str:
    return Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "main"


def get_logger(name: str) -> logging.Logger:
    """Logger named after the module, e.g. 'scenario_engine' rather than 'pipeline.scenario_engine'."""
    configure()
    short = _script_name() if name == "__main__" else name.split(".")[-1]
    return logging.getLogger(short)


def run_main(main) -> None:
    """Run a script's main() with start, duration, and failure logging."""
    name = Path(main.__globals__.get("__file__", "main")).stem
    log = get_logger(name)
    start = time.perf_counter()
    log.info("Started")
    try:
        main()
    except Exception:
        log.exception("Failed after %.1fs", time.perf_counter() - start)
        sys.exit(1)
    log.info("Finished in %.1fs", time.perf_counter() - start)