import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, computed_field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

# --- Sub-Models for better organization ---


class ProjectPaths(BaseModel):
    root: Path
    data: str = "data"
    output: str = "output"
    logs: str = "logs"
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    feature_data: str = "data/features"
    metadata: str = "data/metadata"
    plots: str = "output/ta_plots"
    backtest_results: str = "output/backtest_results"

    def get_abs_path(self, key: str) -> Path:
        relative_path = getattr(self, key)
        return (self.root / relative_path).resolve()


class GlobalSettings(BaseModel):
    symbols: List[str]
    timeframes: List[str]
    resample_map: Dict[str, str]
    debug_mode: bool = Field(alias="runtime.debug_mode", default=True)


class FeatureConfig(BaseModel):
    windows: Dict[str, int]
    hmm_states: int
    tail_risk_sigma: float
    vix_scale_factor: float
    feature_version: str = "v1"


# --- Main Configuration Model ---


class AppConfig(BaseSettings):
    # This class automatically loads environment variables
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Credentials (Loaded from .env automatically via Pydantic)
    alpaca_api_key: str
    alpaca_secret_key: str

    # YAML components
    project: ProjectPaths
    global_settings: GlobalSettings = Field(alias="global")
    data_ingestion: Dict[str, Any]
    feature_engineering: FeatureConfig
    model_training: Dict[str, Any]
    backtest: Dict[str, Any]

    @classmethod
    def load_from_yaml(cls, path: str = "src/config.yaml"):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @property
    def dirs(self):
        """Helper to create and return directory paths on the fly."""
        resolved = {}
        for field in self.project.model_fields:
            if field == "root":
                continue
            path = self.project.get_abs_path(field)
            path.mkdir(parents=True, exist_ok=True)
            resolved[field] = path
        return resolved

    def pretty_print(self):
        """Prints the configuration in a beautiful, formatted panel."""
        console = Console()

        # Convert to dict and mask secrets
        config_dict = self.model_dump()
        # config_dict["alpaca_api_key"] = "********"
        # config_dict["alpaca_secret_key"] = "********"

        # Convert paths to strings for serialization
        def stringify_paths(obj):
            if isinstance(obj, dict):
                return {k: stringify_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [stringify_paths(i) for i in obj]
            return str(obj) if isinstance(obj, Path) else obj

        clean_dict = stringify_paths(config_dict)
        json_data = json.dumps(clean_dict, indent=4)

        # Create a syntax-highlighted block
        syntax = Syntax(json_data, "json", theme="monokai", line_numbers=True)
        console.print(
            Panel(
                syntax,
                title="[bold blue]Application Configuration[/bold blue]",
                expand=False,
            )
        )


config = AppConfig.load_from_yaml()

if __name__ == "__main__":
    # Create a singleton instance

    # 1. Easy access with Autocomplete
    print(f"Project Root: {config.project.root}")
    print(f"Symbols: {config.global_settings.symbols}")

    # 2. Resolved Paths
    print(f"Logs Directory: {config.dirs['logs']}")

    # 3. Secure Dumping
    # Pydantic's model_dump is more powerful than a manual dump function
    # print(config.model_dump(exclude={"alpaca_api_key", "alpaca_secret_key"}))
    config.pretty_print()


# import os
# import yaml
# from pathlib import Path
# from dotenv import load_dotenv
# from typing import Any
# import pprint
# import json


# def _get_nested_dict(data: dict, key: str):
#     """Helper to access nested dictionary keys using dot notation."""
#     keys = key.split(".")
#     for k in keys:
#         data = data.get(k)
#         if data is None:
#             return None
#     return data


# class Config:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Config, cls).__new__(cls)
#         return cls._instance

#     def __init__(self, config_path: str = "src/config.yaml"):
#         if hasattr(self, "_initialized") and self._initialized:
#             return

#         load_dotenv()

#         # Load YAML config
#         with open(config_path, "r") as f:
#             self._config = yaml.safe_load(f)

#         # Project Root
#         self.project_root = Path(self.get("project.root", os.getcwd()))

#         # Credentials
#         self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
#         self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
#         if not self.alpaca_api_key or not self.alpaca_secret_key:
#             raise RuntimeError("Missing Alpaca API credentials in .env file")

#         # Configure paths
#         self._setup_paths()

#         self._initialized = True

#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Retrieves a configuration value using dot notation.
#         Example: config.get('feature_engineering.windows.short')
#         """
#         return _get_nested_dict(self._config, key) or default

#     def _setup_paths(self):
#         """Resolves all paths from config.yaml relative to the project root."""
#         self.paths = {}
#         raw_paths = self.get("paths", {})
#         for key, path_str in raw_paths.items():
#             self.paths[key] = self.project_root / path_str

#         # Create directories
#         for path in self.paths.values():
#             path.mkdir(parents=True, exist_ok=True)

#         # For convenience, expose common paths as attributes
#         self.raw_data_dir = self.paths.get("raw_data")
#         self.processed_data_dir = self.paths.get("processed_data")
#         self.features_dir = self.paths.get("feature_data")
#         self.metadata_dir = self.paths.get("metadata")
#         self.logs_dir = self.paths.get("logs")
#         self.plots_dir = self.paths.get("plots")
#         self.backtest_results_dir = self.paths.get("backtest_results")

#     def __getitem__(self, key: str) -> Any:
#         """Allows dictionary-style access, e.g., config['data_ingestion']"""
#         return self.get(key)

#     def dump(self) -> dict:
#         """Return the entire resolved configuration as a serializable dict."""
#         return {
#             "project_root": str(self.project_root),
#             "yaml_config": self._config,
#             "paths": {k: str(v) for k, v in self.paths.items()},
#             "derived_paths": {
#                 "raw_data_dir": str(self.raw_data_dir),
#                 "processed_data_dir": str(self.processed_data_dir),
#                 "features_dir": str(self.features_dir),
#                 "metadata_dir": str(self.metadata_dir),
#                 "logs_dir": str(self.logs_dir),
#                 "plots_dir": str(self.plots_dir),
#                 "backtest_results_dir": str(self.backtest_results_dir),
#             },
#             "credentials": {
#                 "alpaca_api_key": "*****" if self.alpaca_api_key else None,
#                 "alpaca_secret_key": "*****" if self.alpaca_secret_key else None,
#             },
#         }

#     def dump_json(self) -> str:
#         """
#         Return the entire resolved configuration as a pretty-printed JSON string.
#         Secrets are masked and Paths are stringified.
#         """

#         print("\n" + "=" * 80)
#         print(" FULL APPLICATION CONFIGURATION ".center(80, "="))
#         print("=" * 80)
#         payload = {
#             "project_root": str(self.project_root),
#             "yaml_config": self._config,
#             "paths": {k: str(v) for k, v in self.paths.items()},
#             "derived_paths": {
#                 "raw_data_dir": str(self.raw_data_dir),
#                 "processed_data_dir": str(self.processed_data_dir),
#                 "features_dir": str(self.features_dir),
#                 "metadata_dir": str(self.metadata_dir),
#                 "logs_dir": str(self.logs_dir),
#                 "plots_dir": str(self.plots_dir),
#                 "backtest_results_dir": str(self.backtest_results_dir),
#             },
#             "credentials": {
#                 "alpaca_api_key": "*****" if self.alpaca_api_key else None,
#                 "alpaca_secret_key": "*****" if self.alpaca_secret_key else None,
#             },
#         }
#         print("=" * 80)
#         return json.dumps(payload, indent=2, sort_keys=False)


# if __name__ == "__main__":
#     # Example of how to use the new Config class
#     config = Config()

#     # Access nested value
#     print(f"Short window: {config.get('feature_engineering.windows.short')}")

#     # Access a whole section
#     ingestion_config = config.get("data_ingestion")
#     print(f"Years back for ingestion: {ingestion_config.get('years_back')}")

#     # Access a path
#     print(f"Project Root: {config.project_root}")
#     print(f"Log Path: {config.logs_dir}")
#     print(f"Raw Data Path: {config.raw_data_dir}")

#     print("\n" + "=" * 80)
#     print(" FULL APPLICATION CONFIGURATION ".center(80, "="))
#     print("=" * 80)

#     import pprint

#     pprint.pprint(config.dump(), width=120, sort_dicts=False, indent=2)

#     print("=" * 80)

#     print(config.dump_json())
