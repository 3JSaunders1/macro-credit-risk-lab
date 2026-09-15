import sqlite3
import json
from pathlib import Path
from datetime import datetime

from utils.serialization import to_json_safe

DB_PATH = Path("runs/experiments.db")


class ExperimentStore:

    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp TEXT,
            model_version TEXT,
            params TEXT,
            metrics TEXT,
            artifacts TEXT
        )
        """)
        self.conn.commit()

    def log_run(self, experiment_id, model_version, params, metrics, artifacts):

        # 🔒 FORCE SAFE SERIALIZATION HERE
        params = to_json_safe(params)
        metrics = to_json_safe(metrics)
        artifacts = to_json_safe(artifacts)

        self.conn.execute("""
        INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            datetime.now().isoformat(),
            model_version,
            json.dumps(params),
            json.dumps(metrics),
            json.dumps(artifacts)
        ))

        self.conn.commit()

    def list_runs(self):
        cursor = self.conn.execute("""
        SELECT experiment_id, timestamp, model_version
        FROM experiments
        ORDER BY timestamp DESC
        """)
        return cursor.fetchall()

    def get_run(self, experiment_id):
        cursor = self.conn.execute("""
        SELECT * FROM experiments WHERE experiment_id=?
        """, (experiment_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return {
            "experiment_id": row[0],
            "timestamp": row[1],
            "model_version": row[2],
            "params": json.loads(row[3]),
            "metrics": json.loads(row[4]),
            "artifacts": json.loads(row[5]),
        }


store = ExperimentStore()