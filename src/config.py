import os
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

# -----------------------------
# Sub-Models
# -----------------------------


class ProjectPaths(BaseModel):
    root: Path
    data: str
    output: str
    logs: str
    raw_data: str
    processed_data: str
    corrupted_data: str
    feature_data: str
    metadata: str
    plots: str
    backtest_results: str

    def get_abs_path(self, key: str) -> Path:
        relative_path = getattr(self, key)
        return (self.root / relative_path).resolve()


class GlobalRuntime(BaseModel):
    debug_mode: bool = True


class GlobalSettings(BaseModel):
    symbols: List[str]
    timeframes: List[str]
    resample_map: Dict[str, str]
    runtime: GlobalRuntime


class DataIngestionConfig(BaseModel):
    years_back: int
    chunk_days: int
    adjustment: str
    data_feed: str
    max_requests_per_minute: int


class DataCleaningConfig(BaseModel):
    exchange: str
    ffill_limit: int
    outlier_threshold: float


class FeatureVersioningConfig(BaseModel):
    feature_version: str
    feature_file_template: str


class FeatureEngineeringConfig(BaseModel):
    windows: Dict[str, int]
    hmm_states: int
    tail_risk_sigma: float
    vix_scale_factor: float
    versioning: FeatureVersioningConfig


class ModelTrainingParams(BaseModel):
    objective: str
    eval_metric: str
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    n_estimators: int
    n_jobs: int
    random_state: int


class ModelTrainingConfig(BaseModel):
    train_years: List[int]
    test_years: List[int]
    target_horizon: int
    class_threshold: float
    price_col: str
    model_params: ModelTrainingParams


class ModelOptimizationConfig(BaseModel):
    optimization_years: List[int]
    n_features_to_select: int
    tuner_n_trials: int
    price_col: str
    target_horizon: int
    optimizer_metric: str


class BacktestConfig(BaseModel):
    train_years: List[int]
    test_years: List[int]
    target_horizon: int
    price_col: str
    transaction_cost: float
    allow_shorting: bool
    buy_threshold: float
    sell_threshold: float


# -----------------------------
# Main App Config
# -----------------------------


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    alpaca_api_key: str
    alpaca_secret_key: str

    # YAML-based configs
    project: ProjectPaths
    global_settings: GlobalSettings = Field(alias="global")
    data_ingestion: DataIngestionConfig
    data_cleaning: DataCleaningConfig
    feature_engineering: FeatureEngineeringConfig
    model_training: ModelTrainingConfig
    model_optimization: ModelOptimizationConfig
    backtest: BacktestConfig

    @classmethod
    def load_from_yaml(cls, path: str = "src/config.yaml"):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @property
    def dirs(self) -> Dict[str, Path]:
        """Return all resolved directories as Path objects, creating them if needed."""
        resolved = {}
        for field in self.project.model_fields:
            if field == "root":
                continue
            path = self.project.get_abs_path(field)
            path.mkdir(parents=True, exist_ok=True)
            resolved[field] = path
        return resolved

    def pretty_print(self):
        """Print configuration in rich JSON format."""
        console = Console()
        config_dict = self.model_dump()

        def stringify_paths(obj):
            if isinstance(obj, dict):
                return {k: stringify_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [stringify_paths(i) for i in obj]
            return str(obj) if isinstance(obj, Path) else obj

        clean_dict = stringify_paths(config_dict)
        json_data = json.dumps(clean_dict, indent=4)
        syntax = Syntax(json_data, "json", theme="monokai", line_numbers=True)
        console.print(
            Panel(
                syntax,
                title="[bold blue]Application Configuration[/bold blue]",
                expand=False,
            )
        )


# -----------------------------
# Singleton instance
# -----------------------------
config = AppConfig.load_from_yaml()


if __name__ == "__main__":
    print(f"Project Root: {config.project.root}")
    print(f"Symbols: {config.global_settings.symbols}")
    print(f"Logs Directory: {config.dirs['logs']}")
    config.pretty_print()
