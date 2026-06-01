"""Typed configuration loader.

Every threshold, fee, and limit the pipeline uses is read from
config/settings.yaml through this module. The guardrail is simple: no magic
numbers scattered through the code, read them here.

This is the offline-research subset of the configuration. The live-trading
sections of the full project (venue endpoints, API-wallet secrets, the
kill-switch, execution mode) are not part of this demo and are omitted.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------- paths

# This file lives at <repo>/xortcut/config.py, so the repo root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ----------------------------------------------------------------- models
# extra="allow" so new keys added to settings.yaml never break loading.


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class Symbols(_Base):
    core: List[str] = Field(default_factory=lambda: ["BTC", "ETH"])


class Risk(_Base):
    # The only risk setting the cost model reads: maker vs taker fee selection.
    prefer_maker: bool = True


class Labeling(_Base):
    method: str = "triple_barrier"
    up_mult: float = 1.5                  # upper barrier = price * (1 + up_mult * vol)
    dn_mult: float = 1.5                  # lower barrier = price * (1 - dn_mult * vol)
    max_holding_bars: int = 24            # vertical (time) barrier, in bars
    vol_window: int = 50                  # bars used to estimate recent volatility


class Features(_Base):
    technical: bool = True                # family A: commodity, table stakes
    microstructure: bool = True           # family B: the differentiator
    context: bool = True                  # family C: realized vol, session, BTC-as-feature


class Backtest(_Base):
    # Stored as percentages, e.g. 0.045 means 0.045 percent. Convert to a
    # fraction with the fraction helpers below before applying to a notional.
    fee_taker_pct: float = 0.045
    fee_maker_pct: float = 0.015
    funding_settlement: str = "hourly"
    slippage_model: str = "spread_plus_impact"
    slippage_bps: float = 3

    @property
    def fee_taker_frac(self) -> float:
        return self.fee_taker_pct / 100.0

    @property
    def fee_maker_frac(self) -> float:
        return self.fee_maker_pct / 100.0

    @property
    def slippage_frac(self) -> float:
        return self.slippage_bps / 10_000.0


class WalkForward(_Base):
    train_days: int = 180
    test_days: int = 30


class Cpcv(_Base):
    n_groups: int = 6                     # combinatorial purged cross-validation groups
    embargo_bars: int = 24                # embargo around each test fold


class Acceptance(_Base):
    min_oos_sharpe: float = 0.8           # net of costs, out-of-sample
    max_pbo: float = 0.40                 # probability of backtest overfitting, lower is better
    max_is_oos_degradation_pct: float = 40


class Validation(_Base):
    walk_forward: WalkForward = Field(default_factory=WalkForward)
    cpcv: Cpcv = Field(default_factory=Cpcv)
    purge: bool = True
    acceptance: Acceptance = Field(default_factory=Acceptance)


class Data(_Base):
    intervals: List[str] = Field(default_factory=lambda: ["15m", "1h"])
    history_days: int = 365
    storage_path: str = "data/parquet"
    timezone: str = "UTC"

    def storage_dir(self, root: Optional[Path] = None) -> Path:
        base = Path(root) if root is not None else PROJECT_ROOT
        p = Path(self.storage_path)
        return p if p.is_absolute() else base / p


class Project(_Base):
    name: str = "xortcut-backtest-lab"


class Settings(_Base):
    project: Project = Field(default_factory=Project)
    symbols: Symbols = Field(default_factory=Symbols)
    risk: Risk = Field(default_factory=Risk)
    labeling: Labeling = Field(default_factory=Labeling)
    features: Features = Field(default_factory=Features)
    backtest: Backtest = Field(default_factory=Backtest)
    validation: Validation = Field(default_factory=Validation)
    data: Data = Field(default_factory=Data)


# ----------------------------------------------------------------- loader


@lru_cache(maxsize=8)
def load_settings(path: Optional[str] = None) -> Settings:
    """Load and validate config/settings.yaml. Cached per path."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Settings file not found at {settings_path}. Expected config/settings.yaml."
        )
    raw = yaml.safe_load(settings_path.read_text()) or {}
    return Settings(**raw)
