from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle


class QualityAssessmentPlots:
    def __init__(self, plots_dir: Path, logger):
        self.plots_dir = plots_dir
        self.logger = logger
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_all_plots(self, assessment_results: dict):
        """Generate all plots based on dictionary results"""

        # Temporal coverage
        df = assessment_results.get("temporal_coverage")
        if df is not None and not df.empty:
            self.plot_temporal_coverage(df)

        # Intraday completeness
        df = assessment_results.get("intraday_completeness")
        if df is not None and not df.empty:
            self.plot_bar_count_distribution(df)

        # OHLC integrity
        df = assessment_results.get("ohlc_integrity")
        if df is not None and not df.empty:
            self.plot_ohlc_violations(df)

        # Price anomalies
        df_ranges = assessment_results.get("price_anomalies", {}).get("price_ranges")
        df_shifts = assessment_results.get("price_anomalies", {}).get(
            "suspicious_shifts"
        )
        if df_ranges is not None and not df_ranges.empty:
            self.plot_price_ranges(df_ranges)
        if df_shifts is not None and not df_shifts.empty:
            self.plot_price_shifts(df_shifts)

        # Volume stats
        df = assessment_results.get("volume_stats")
        if df is not None and not df.empty:
            self.plot_volume_distribution(df)

        # Market concordance
        df = assessment_results.get("market_concordance")
        if df is not None and not df.empty:
            self.plot_market_concordance(df)

        # Synchronized availability
        df = assessment_results.get("synchronized_availability")
        if df is not None and not df.empty:
            self.plot_synchronized_availability(df)

    # --- Plots for each assessment ---

    def plot_temporal_coverage(self, df_coverage):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_coverage, x="symbol", y="coverage_pct", palette="viridis")
        plt.title("1.2.1 Temporal Coverage % by Symbol")
        plt.ylim(0, 1.1)
        plt.axhline(1.0, color="r", linestyle="--", label="100% Expected")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        self._save("1.2.1_temporal_coverage.png")

    def plot_bar_count_distribution(self, df_counts):
        plt.figure(figsize=(12, 6))
        data = df_counts[
            (df_counts["bar_count"] > 100) & (df_counts["bar_count"] < 500)
        ]
        sns.histplot(data=data, x="bar_count", bins=50, kde=False)
        plt.axvline(390, color="r", linestyle="--", label="Standard (390)")
        plt.axvline(210, color="orange", linestyle="--", label="Early Close (210)")
        plt.title("1.2.2 Intraday Bar Count Distribution")
        plt.xlabel("Bars per Day")
        plt.legend()
        plt.tight_layout()
        self._save("1.2.2_bar_count_dist.png")

    def plot_ohlc_violations(self, df_violations):
        if df_violations["count"].sum() == 0:
            self.logger.info("No OHLC violations to plot.")
            return
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_violations, x="violation_type", y="count", palette="magma")
        plt.title("1.3.1 OHLC Schema Violations")
        plt.yscale("log")
        plt.ylabel("Count (Log Scale)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save("1.3.1_ohlc_violations.png")

    def plot_price_ranges(self, df_ranges):
        plt.figure(figsize=(12, 6))
        df_ranges = df_ranges.sort_values("max", ascending=False)
        plt.bar(
            df_ranges["symbol"],
            df_ranges["max"] - df_ranges["min"],
            bottom=df_ranges["min"],
            color="skyblue",
            edgecolor="blue",
        )
        plt.title("1.4.1 Price Range (Min to Max)")
        plt.ylabel("Price ($)")
        plt.xticks(rotation=90)
        plt.yscale("log")
        plt.tight_layout()
        self._save("1.4.1_price_ranges.png")

    def plot_price_shifts(self, df_shifts):
        if df_shifts.empty:
            self.logger.info("No suspicious price shifts to plot.")
            return
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=df_shifts,
            x="date",
            y="close",
            hue="symbol",
            palette="tab10",
            s=50,
        )
        plt.title("1.4.1 Suspicious Daily Price Shifts (>50%)")
        plt.ylabel("Close Price ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save("1.4.1_price_suspicious_shifts.png")

    def plot_volume_distribution(self, df_vol):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_vol, x="symbol", y="mean_volume", palette="Blues_d")
        plt.title("1.4.2 Mean Daily Volume")
        plt.yscale("log")
        plt.xticks(rotation=90)
        plt.tight_layout()
        self._save("1.4.2_volume_stats.png")

    def plot_market_concordance(self, df_conc):
        times = [t.strftime("%H:%M") for t in df_conc.index]
        x_ticks = np.arange(0, len(times), 30)
        plt.figure(figsize=(14, 7))
        plt.stackplot(
            range(len(times)),
            df_conc["pct_up"],
            df_conc["pct_flat"],
            df_conc["pct_down"],
            labels=["Up", "Flat", "Down"],
            colors=["green", "gray", "red"],
            alpha=0.6,
        )
        plt.xticks(x_ticks, [times[i] for i in x_ticks], rotation=45)
        plt.title("1.5.2 Intraday Market Concordance")
        plt.xlabel("Time of Day")
        plt.ylabel("Proportion of Universe")
        plt.legend(loc="upper left")
        plt.tight_layout()
        self._save("1.5.2_market_concordance.png")

    def plot_synchronized_availability(self, df_sync):
        plt.figure(figsize=(12, 6))
        df_sync.plot(y="coverage_pct", kind="bar", color="teal")
        plt.title("1.5.1 Synchronized Symbol Availability %")
        plt.ylabel("Availability (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save("1.5.1_synchronized_availability.png")

    def _save(self, filename):
        path = self.plots_dir / filename
        plt.savefig(path, dpi=150)
        self.logger.info(f"Saved plot: {path}")
        plt.close()
