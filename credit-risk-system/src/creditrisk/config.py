"""Typed configuration loaded from config.yaml (no hardcoded paths anywhere else)."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_NAME = "config.yaml"
ENV_VAR = "CREDITRISK_CONFIG"


class Paths(BaseModel):
    kaggle_file: Path
    synthetic_file: Path
    models_dir: Path
    reports_dir: Path
    database: Path


class DataCfg(BaseModel):
    synthetic_rows: int = 60000
    synthetic_default_rate: float = 0.07


class SplitCfg(BaseModel):
    test_size: float = 0.2
    validation_size: float = 0.2


class CleaningCfg(BaseModel):
    utilization_cap: float = 1.5
    debt_ratio_cap: float = 5.0
    past_due_cap: int = 10
    monthly_income_cap: float = 50000
    open_credit_lines_cap: int = 30
    real_estate_loans_cap: int = 10
    dependents_cap: int = 10
    dependents_fill: int = 0
    age_min: int = 18
    age_max: int = 100


class FeaturesCfg(BaseModel):
    high_debt_ratio_flag: float = 0.43


class ScorecardCfg(BaseModel):
    max_bins: int = 8
    min_bin_share: float = 0.05
    min_iv: float = 0.02


class ScoreCfg(BaseModel):
    base_score: float = 600
    base_odds: float = 50
    pdo: float = 20


class ModelCfg(BaseModel):
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    min_samples_leaf: int = 50


class CostsCfg(BaseModel):
    cost_of_default: float = 10000
    lost_profit_good: float = 1000
    refer_ratio: float = Field(0.4, gt=0, lt=1)


class ThresholdsCfg(BaseModel):
    approve_below: float | None = None
    decline_at_or_above: float | None = None


class RiskBandsCfg(BaseModel):
    low_below: float = 0.03
    high_at_or_above: float = 0.10


class DriftCfg(BaseModel):
    warn: float = 0.10
    alert: float = 0.25
    min_rows: int = 100


class ReasonsCfg(BaseModel):
    max_reasons: int = 4


class Settings(BaseModel):
    seed: int = 42
    paths: Paths
    data: DataCfg = DataCfg()
    split: SplitCfg = SplitCfg()
    cleaning: CleaningCfg = CleaningCfg()
    features: FeaturesCfg = FeaturesCfg()
    scorecard: ScorecardCfg = ScorecardCfg()
    score: ScoreCfg = ScoreCfg()
    model: ModelCfg = ModelCfg()
    costs: CostsCfg = CostsCfg()
    thresholds: ThresholdsCfg = ThresholdsCfg()
    risk_bands: RiskBandsCfg = RiskBandsCfg()
    drift: DriftCfg = DriftCfg()
    reasons: ReasonsCfg = ReasonsCfg()


def project_root() -> Path:
    """Project root: the folder that contains config.yaml (two levels above src/creditrisk)."""
    return Path(__file__).resolve().parents[2]


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings; relative paths in the file are resolved against the config file's folder."""
    cfg_path = Path(path or os.environ.get(ENV_VAR) or project_root() / DEFAULT_CONFIG_NAME).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    raw["paths"] = {k: str((base / v).resolve()) for k, v in raw["paths"].items()}
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def setup_logging(level: int = logging.INFO) -> None:
    """Console logging for scripts (library modules only use logging.getLogger)."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
