# config/settings.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    DATA_PATH: str = "data/macro_data.csv"

    # VAR
    MAX_VAR_LAGS: int = 8
    VAR_LAGS: Optional[int] = 2

    # Confidence intervals
    STRESS_ALPHA: float = 0.10

    # Credit model betas (calibrated priors)
    BETA_U: float = 0.18
    BETA_PI: float = 0.10
    BETA_0: float = -6.0

    # IRF horizon
    IRF_HORIZON: int = 12

    # Shock engine
    DEFAULT_SHOCK_SIZE: float = 1.0
    DEFAULT_SHOCK_HORIZON: int = 8

    # BVAR Minnesota prior
    BVAR_LAMBDA: float = 0.2   # tightness
    BVAR_LAGS: int = 2

    # Local projections
    LP_HORIZON: int = 12
    LP_LAGS: int = 2


settings = Settings()