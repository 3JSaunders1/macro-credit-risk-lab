"""
utils/run_metadata.py
---------------------
Now stores full experiment metadata for:
- run history dashboard
- model comparison
- research reproducibility
"""

import json
from datetime import datetime
from pathlib import Path


RUN_DIR = Path("runs")


def save_run_metadata(result: dict):
    RUN_DIR.mkdir(exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": result.get("experiment_id"),
        "model_version": result.get("model_version"),

        "pd": result.get("predicted_pd"),
        "rating": result.get("rating"),

        "forecast": result.get("forecast"),
    }

    file_path = RUN_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    return str(file_path)