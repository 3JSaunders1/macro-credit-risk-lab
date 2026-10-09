"""
utils/run_metadata.py
---------------------
Save a small JSON record of a CLI run (forecast, illustrative PD, and rating).
Written only on request, to reports/cli_runs/ under the project root.
"""
import json
from datetime import datetime
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "reports" / "cli_runs"


def save_run_metadata(result: dict) -> str:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    payload = {
        "timestamp": now.isoformat(),
        "experiment_id": result.get("experiment_id"),
        "model_type": result.get("model_type"),
        "pd": result.get("predicted_pd"),
        "rating": result.get("rating"),
        "forecast": result.get("forecast"),
    }
    file_path = RUN_DIR / f"run_{now.strftime('%Y%m%d_%H%M%S')}.json"
    file_path.write_text(json.dumps(payload, indent=2))
    return str(file_path)