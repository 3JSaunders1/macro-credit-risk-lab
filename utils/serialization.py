# utils/serialization.py

import numpy as np
import pandas as pd
from datetime import datetime


def to_json_safe(obj):
    """
    Recursively convert anything into JSON-safe Python primitives.
    This is the SINGLE source of truth for all logging + storage.
    """

    # NumPy scalars (int64, float64, etc.)
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # Datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Dict
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    return obj