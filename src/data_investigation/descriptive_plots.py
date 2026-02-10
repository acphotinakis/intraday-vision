import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


class DescriptivePlots:
    def __init__(self, plots_dir: Path, logger):
        self.plots_dir = plots_dir
        self.logger = logger
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper")

    def generate_all_plots(self, results: dict):
        # 2.2 Price Dynamics
        if not results["price_dynamics"]["cumulative_returns"].empty:
            self.plot_cumulative_returns(
                results["price_dynamics"]["cumulative_returns"]
            )

        if "profile" in results["price_dynamics"]["intraday_patterns"]:
            self.plot_intraday_returns(
                results["price_dynamics"]["intraday_patterns"]["profile"]
            )
            self.plot_intraday_ttest(
                results["price_dynamics"]["intraday_patterns"]["stats"]
            )

        self.plot_distribution_stats(results["price_dynamics"]["distribution_stats"])

        # 2.3 Volatility
        self.plot_realized_volatility(results["volatility"]["realized_vol"])
        self.plot_volatility_clustering(results["volatility"]["vol_clustering"])
        self.plot_intraday_vol_surface(results["volatility"]["intraday_vol"])

        # 2.4 Volume
        self.plot_volume_profile(results["volume"]["intraday_profile"])
        self.plot_vol_vol_relationship(results["volume"]["vol_vol_relation"])

        # 2.5 Cross-Sectional
        self.plot_correlation_matrix(results["cross_sectional"]["correlation"])
        self.plot_rolling_betas(results["cross_sectional"]["betas"])

    # --- 2.2 Price Dynamics ---
    def plot_cumulative_returns(self, df):
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            lw = 2.5 if col in ["NVDA", "TSLA", "AAPL"] else 1.0
            alpha = 1.0 if col in ["NVDA", "TSLA", "AAPL"] else 0.6
            plt.plot(df.index, df[col], label=col, linewidth=lw, alpha=alpha)
        plt.title("2.2.1 Normalized Price Evolution (Base=100)")
        plt.ylabel("Cumulative Return")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self._save("2.2.1_cumulative_returns.png")

    def plot_intraday_returns(self, series):
        plt.figure(figsize=(10, 5))
        times = [t.strftime("%H:%M") for t in series.index]
        x_ticks = np.arange(0, len(times), 30)
        plt.plot(range(len(times)), series.values, color="purple", linewidth=2)
        plt.xticks(x_ticks, [times[i] for i in x_ticks], rotation=45)
        plt.title("2.2.2 Intraday Mean Returns")
        plt.ylabel("Return (bps)")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        self._save("2.2.2_intraday_returns.png")

    def plot_intraday_ttest(self, stats_dict):
        plt.figure(figsize=(6, 4))
        plt.bar(
            ["t_stat", "p_value"],
            [stats_dict["t_stat"], stats_dict["p_value"]],
            color=["orange", "green"],
        )
        plt.title("2.2.2 T-Test Morning vs Evening Returns")
        self._save("2.2.2_intraday_ttest.png")

    def plot_distribution_stats(self, df_stats):
        plt.figure(figsize=(10, 6))
        df_plot = df_stats.reset_index()
        sns.barplot(x="symbol", y="kurtosis", data=df_plot, palette="Reds")
        plt.axhline(3.0, color="blue", linestyle="--", label="Gaussian")
        plt.title("2.2.3 Return Distribution (Kurtosis)")
        plt.ylabel("Kurtosis")
        plt.legend()
        self._save("2.2.3_distribution_stats.png")

    # --- 2.3 Volatility ---
    def plot_realized_volatility(self, series):
        plt.figure(figsize=(12, 6))
        series.plot()
        plt.title("2.3.1 Rolling Realized Volatility")
        plt.ylabel("Volatility (Annualized)")
        self._save("2.3.1_realized_volatility.png")

    def plot_volatility_clustering(self, df_acf):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_acf, cmap="Blues", annot=True, fmt=".2f")
        plt.title("2.3.2 Volatility Autocorrelation (ACF)")
        plt.xlabel("Symbol")
        plt.ylabel("Lag (Minutes)")
        self._save("2.3.2_vol_clustering.png")

    def plot_intraday_vol_surface(self, df_surface):
        plt.figure(figsize=(14, 8))
        df_plot = df_surface.copy()
        df_plot.columns = [t.strftime("%H:%M") for t in df_plot.columns]
        if len(df_plot.columns) > 50:
            df_plot = df_plot.iloc[:, ::15]
        sns.heatmap(df_plot, cmap="coolwarm", cbar_kws={"label": "Annualized Vol"})
        plt.title("2.3.3 Intraday Volatility Surface")
        plt.xlabel("Time of Day")
        plt.ylabel("Symbol")
        self._save("2.3.3_intraday_vol_surface.png")

    # --- 2.4 Volume ---
    def plot_volume_profile(self, series):
        plt.figure(figsize=(10, 5))
        times = [t.strftime("%H:%M") for t in series.index]
        x_ticks = np.arange(0, len(times), 30)
        plt.fill_between(range(len(times)), series.values, color="skyblue", alpha=0.4)
        plt.plot(range(len(times)), series.values, color="navy")
        plt.xticks(x_ticks, [times[i] for i in x_ticks], rotation=45)
        plt.title("2.4.1 Intraday Volume Profile")
        plt.ylabel("Normalized Volume")
        self._save("2.4.1_volume_profile.png")

    def plot_vol_vol_relationship(self, stats_dict):
        plt.figure(figsize=(6, 4))
        keys = ["slope", "intercept", "r_squared"]
        values = [stats_dict[k] for k in keys]
        plt.bar(keys, values, color=["purple", "orange", "green"])
        plt.title("2.4.2 Volume-Volatility Regression Stats")
        self._save("2.4.2_vol_vol_relationship.png")

    # --- 2.5 Cross-Sectional ---
    def plot_correlation_matrix(self, df_corr):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, cmap="vlag", center=0, vmin=-1, vmax=1)
        plt.title("2.5.1 Correlation Matrix")
        self._save("2.5.1_correlation_matrix.png")

    def plot_rolling_betas(self, df_betas):
        plt.figure(figsize=(12, 6))
        subset = df_betas.iloc[:, :5]
        subset.plot(ax=plt.gca())
        plt.axhline(
            1.0, color="black", linestyle="--", linewidth=0.8, label="Market (1.0)"
        )
        plt.title("2.5.2 Rolling Beta")
        plt.ylabel("Beta")
        plt.legend()
        self._save("2.5.2_rolling_betas.png")

    def _save(self, filename):
        path = self.plots_dir / filename
        plt.savefig(path, dpi=150)
        self.logger.info(f"Saved plot: {path}")
        plt.close()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from scipy import stats


# class DescriptivePlots:
#     def __init__(self, plots_dir: Path, logger):
#         self.plots_dir = plots_dir
#         self.logger = logger
#         self.plots_dir.mkdir(parents=True, exist_ok=True)
#         sns.set_theme(style="whitegrid", context="paper")

#     def generate_all_plots(self, results: dict):
#         # 2.2 Price Dynamics
#         if not results["price_dynamics"]["cumulative_returns"].empty:
#             self.plot_normalized_price_paths(
#                 results["price_dynamics"]["cumulative_returns"]
#             )

#         if not results["price_dynamics"]["intraday_patterns"]["profile"].empty:
#             self.plot_intraday_seasonality(
#                 results["price_dynamics"]["intraday_patterns"]["profile"]
#             )

#         self.plot_return_distribution(results["price_dynamics"]["distribution_stats"])

#         # 2.3 Volatility
#         self.plot_volatility_clustering(results["volatility"]["vol_clustering"])
#         self.plot_volatility_surface(results["volatility"]["intraday_vol"])

#         # 2.4 Volume
#         self.plot_volume_profile(results["volume"]["intraday_profile"])

#         # 2.5 Cross-Sectional
#         self.plot_correlation_matrix(results["cross_sectional"]["correlation"])
#         self.plot_rolling_beta(results["cross_sectional"]["betas"])

#     def plot_normalized_price_paths(self, df_norm):
#         """2.2.1 Normalized Price Evolution"""
#         plt.figure(figsize=(12, 6))
#         #
#         for col in df_norm.columns:
#             # Highlight Tech giants vs others if needed, for now plot all
#             linewidth = 2.5 if col in ["NVDA", "TSLA", "AAPL"] else 1.0
#             alpha = 1.0 if col in ["NVDA", "TSLA", "AAPL"] else 0.6
#             plt.plot(
#                 df_norm.index, df_norm[col], label=col, linewidth=linewidth, alpha=alpha
#             )

#         plt.title("2.2.1 Normalized Price Evolution (Base=100)")
#         plt.ylabel("Cumulative Return (Normalized)")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         self._save("2.2.1_price_evolution.png")

#     def plot_intraday_seasonality(self, series_profile):
#         """2.2.2 Intraday U-Shape"""
#         plt.figure(figsize=(10, 5))
#         #
#         # Convert index (time objects) to string for plotting
#         times = [t.strftime("%H:%M") for t in series_profile.index]
#         x_ticks = np.arange(0, len(times), 30)

#         plt.plot(range(len(times)), series_profile.values, color="purple", linewidth=2)
#         plt.xticks(x_ticks, [times[i] for i in x_ticks], rotation=45)
#         plt.title("2.2.2 Average Intraday Return Profile (Minute by Minute)")
#         plt.ylabel("Avg Return (bps)")
#         plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
#         self._save("2.2.2_intraday_seasonality.png")

#     def plot_return_distribution(self, df_stats):
#         """2.2.3 Return Distribution vs Normal"""
#         plt.figure(figsize=(10, 6))
#         df_plot = df_stats.reset_index()
#         sns.barplot(x="symbol", y="kurtosis", data=df_plot, palette="Reds")
#         plt.axhline(3.0, color="blue", linestyle="--", label="Gaussian (3.0)")
#         plt.title("2.2.3 Excess Kurtosis (Fat Tails Check)")
#         plt.ylabel("Kurtosis")
#         plt.legend()
#         self._save("2.2.3_kurtosis_check.png")

#     def plot_volatility_clustering(self, df_acf):
#         """2.3.2 Volatility Clustering (ACF)"""
#         plt.figure(figsize=(10, 6))
#         #
#         # Transpose so Lags are on X axis
#         sns.heatmap(df_acf, cmap="Blues", annot=True, fmt=".2f")
#         plt.title("2.3.2 Volatility Autocorrelation (Clustering Evidence)")
#         plt.xlabel("Symbol")
#         plt.ylabel("Lag (Minutes)")
#         self._save("2.3.2_vol_clustering.png")

#     def plot_volatility_surface(self, df_surface):
#         """2.3.3 Intraday Volatility Heatmap"""
#         plt.figure(figsize=(14, 8))
#         #
#         # We need to process the index (Time) to be readable
#         df_plot = df_surface.copy()
#         df_plot.columns = [t.strftime("%H:%M") for t in df_plot.columns]

#         # Downsample columns for heatmap readability if 1-min resolution
#         if len(df_plot.columns) > 50:
#             df_plot = df_plot.iloc[:, ::15]  # Every 15th minute

#         sns.heatmap(
#             df_plot, cmap="coolwarm", cbar_kws={"label": "Annualized Volatility"}
#         )
#         plt.title("2.3.3 Intraday Volatility Surface")
#         plt.xlabel("Time of Day")
#         plt.ylabel("Symbol")
#         self._save("2.3.3_vol_surface.png")

#     def plot_volume_profile(self, series_vol):
#         """2.4.1 Intraday Volume Profile"""
#         plt.figure(figsize=(10, 5))

#         times = [t.strftime("%H:%M") for t in series_vol.index]
#         x_ticks = np.arange(0, len(times), 30)

#         plt.fill_between(
#             range(len(times)), series_vol.values, color="skyblue", alpha=0.4
#         )
#         plt.plot(range(len(times)), series_vol.values, color="navy")

#         plt.xticks(x_ticks, [times[i] for i in x_ticks], rotation=45)
#         plt.title("2.4.1 Average Intraday Volume Profile")
#         plt.ylabel("Relative Volume (1.0 = Daily Mean)")
#         self._save("2.4.1_volume_profile.png")

#     def plot_correlation_matrix(self, df_corr):
#         """2.5.1 Correlation Matrix"""
#         plt.figure(figsize=(10, 8))
#         #
#         sns.heatmap(df_corr, annot=True, cmap="vlag", center=0, vmin=-1, vmax=1)
#         plt.title("2.5.1 Return Correlation Matrix")
#         self._save("2.5.1_correlation_matrix.png")

#     def plot_rolling_beta(self, df_betas):
#         """2.5.2 Rolling Betas"""
#         plt.figure(figsize=(12, 6))
#         #
#         # Plot only a few key symbols to avoid clutter
#         subset = df_betas.iloc[:, :5]  # First 5 symbols
#         subset.plot(ax=plt.gca())

#         plt.axhline(
#             1.0, color="black", linestyle="--", linewidth=0.8, label="Market (1.0)"
#         )
#         plt.title("2.5.2 Rolling 20-Day Beta")
#         plt.ylabel("Beta to Benchmark")
#         plt.legend()
#         self._save("2.5.2_rolling_beta.png")

#     def _save(self, filename):
#         path = self.plots_dir / filename
#         plt.savefig(path, dpi=150)
#         self.logger.info(f"Saved plot: {path}")
#         plt.close()
