# ==============================================================================
#                 LIQUID VEINS - ALGORITHMIC TRADING PIPELINE
# ==============================================================================
#
# This Makefile orchestrates the ML pipeline using a central Python orchestrator.
#
.PHONY: all help setup clean data model full-eval run ingest clean_data features optimize train backtest

# --- Variables ---
PYTHON := python3
ARGS := "" # Default to empty ARGS

# --- Core Commands ---

help:
	@echo "==========================================================================="
	@echo "                     Stock Distro Pipeline Manager                     "
	@echo "==========================================================================="
	@echo "Usage:"
	@echo "  make <target> [ARGS=\"...\" | P=\"...\"]"
	@echo ""
	@echo "Primary Targets:"
	@echo "  all          - Run the entire pipeline end-to-end."
	@echo "  data         - Run the data pipeline (ingest, clean)."
	@echo "  model        - Run the model generation pipeline (features, train, optimize)."
	@echo "  full-eval    - Run a full evaluation from features to backtest."
	@echo "  run          - Run specific pipelines. Pass pipeline names via ARGS."
	@echo "               Example: make run ARGS=\"ingest clean features\""
	@echo ""
	@echo "Individual Pipeline Stages:"
	@echo "  ingest       - Download raw data from Alpaca."
	@echo "  clean_data   - Clean and verify raw data."
	@echo "  features     - Generate features from cleaned data."
	@echo "  optimize     - Run feature selection and hyperparameter tuning."
	@echo "  train        - Train the final model on the training set."
	@echo "  backtest     - Run the strategy backtest on the test set."
	@echo ""
	@echo "Maintenance:"
	@echo "  setup        - Install all Python dependencies from requirements.txt."
	@echo "  clean        - Remove all generated data files (processed and features)."
	@echo "==========================================================================="

# --- Meta Targets (Workflows) ---

all:
	@echo "🚀 Starting full end-to-end pipeline..."
	$(PYTHON) -m src.main run-all
	@echo "✅ Full pipeline execution complete."

data:
	@echo "📊 Starting data workflow (ingest, clean)..."
	$(MAKE) run ARGS="ingest clean"

model:
	@echo "🤖 Starting model generation workflow (features, train, optimize)..."
	$(MAKE) run ARGS="features train optimize"

full-eval:
	@echo "📈 Starting full evaluation workflow (features, backtest)..."
	$(MAKE) run ARGS="features backtest"


# --- Dynamic Runner for Individual Stages ---
# This allows running `make ingest`, `make clean`, etc. as shortcuts
ingest: ARGS=ingest
clean_data: ARGS=clean
features: ARGS=features
optimize: ARGS=optimize
train: ARGS=train
backtest: ARGS=backtest
ingest clean_data features optimize train backtest: run

run:
	@echo "🏃 Running pipeline(s): $(ARGS)..."
	$(PYTHON) -m src.main run $(ARGS)

# --- Maintenance Targets ---

setup:
	@echo "📦 Installing dependencies from requirements.txt..."
	@$(PYTHON) -m pip install -r requirements.txt

clean:
	@echo "🔥 Deleting generated data..."
	@echo "   - Deleting processed data..."
	@find data/processed -name "*.parquet" -delete || echo "No processed files to delete."
	@echo "   - Deleting feature data..."
	@find data/features -name "*.parquet" -delete || echo "No feature files to delete."
	@echo "   - Deleting old plot files"
	@find . -type f -name 'plot_*.png' -delete
	@echo "✅ Cleanup complete."
