import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from models.var_model import VARModel

ARTIFACT_DIR = Path("models/artifacts")
REGISTRY_FILE = ARTIFACT_DIR / "registry.pkl"


@dataclass
class ModelBundle:
    version: str
    experiment_id: str
    timestamp: str

    var_model: VARModel
    irf: object
    fevd: object


class ModelRegistry:

    def __init__(self):
        self.registry = self._load_registry()

    def _load_registry(self):
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_registry(self):
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, "wb") as f:
            pickle.dump(self.registry, f)

    def create_and_train(self, data, version: str = None):

        experiment_id = str(uuid.uuid4())[:8]
        version = version or f"v{len(self.registry) + 1}"
        timestamp = datetime.now().isoformat()

        model = VARModel()
        model.fit(data)

        irf_bundle = model.irf(10)

        bundle = ModelBundle(
            version=version,
            experiment_id=experiment_id,
            timestamp=timestamp,
            var_model=model,
            irf=irf_bundle.irfs,
            fevd=irf_bundle.fevd,
        )

        self.registry[experiment_id] = bundle
        self._save_registry()

        return bundle

    def load_latest(self):
        if not self.registry:
            raise ValueError("Registry empty — run training mode first")

        latest_id = sorted(self.registry.keys())[-1]
        return self.registry[latest_id]

    def get(self, experiment_id: str):
        return self.registry[experiment_id]

    def list_runs(self):
        return [
            {
                "experiment_id": k,
                "version": v.version,
                "timestamp": v.timestamp,
            }
            for k, v in self.registry.items()
        ]


registry = ModelRegistry()