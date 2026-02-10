import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class FeaturePlots:
    def __init__(self, plots_dir: Path, logger):
        self.plots_dir = plots_dir
        self.logger = logger
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_all_plots(self, analysis_results: dict):
        self.plot_autocorrelation_heatmap(analysis_results["autocorrelation"])
        self.plot_asymmetric_response(analysis_results["asymmetric_response"])
        self.plot_feature_correlation(analysis_results["correlation_matrix"])
        self.plot_pca_variance(analysis_results["pca_analysis"])
        self.plot_predictive_power(analysis_results["predictive_power"])
        self.plot_feature_stability(analysis_results["feature_stability"])

    def plot_autocorrelation_heatmap(self, df_ac):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_ac, annot=True, cmap="coolwarm", center=0)
        plt.title("3.2.1 Autocorrelation of Returns by Horizon")
        self._save("3.2.1_autocorrelation_heatmap.png")

    def plot_asymmetric_response(self, result):
        """3.2.2 Asymmetric Response Visualization"""
        if not result or result["contingency_table"] is None:
            return

        ct = result["contingency_table"]
        # Convert to percentages row-wise
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            ct_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar_kws={"label": "Probability (%)"},
        )
        plt.title(f"3.2.2 Asymmetric Response (p={result['p_value']:.4f})")
        plt.ylabel("Current Bar Direction")
        plt.xlabel("Next Bar Direction")
        self._save("3.2.2_asymmetric_response.png")

    def plot_feature_correlation(self, df_corr):
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        sns.heatmap(
            df_corr,
            mask=mask,
            cmap="vlag",
            center=0,
            square=True,
            linewidths=0.5,
            annot=False,
        )
        plt.title("3.6.1 Feature Correlation Matrix")
        plt.tight_layout()
        self._save("3.6.1_feature_correlation.png")

    def plot_pca_variance(self, pca_results):
        if not pca_results:
            return
        exp_var = pca_results["explained_variance"] * 100
        cum_var = pca_results["cumulative_variance"] * 100
        n = len(exp_var)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(range(1, n + 1), exp_var, alpha=0.6, label="Individual")
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance (%)")

        ax2 = ax1.twinx()
        ax2.plot(range(1, n + 1), cum_var, color="red", marker="o", label="Cumulative")
        ax2.set_ylim(0, 105)

        plt.title("3.6.2 PCA Explained Variance")
        self._save("3.6.2_pca_variance.png")

    def plot_predictive_power(self, df_pred):
        plt.figure(figsize=(12, 8))
        df_sorted = df_pred.sort_values("r_squared", ascending=True)
        bars = plt.barh(df_sorted["feature"], df_sorted["r_squared"], color="teal")
        plt.xlabel("R-Squared")
        plt.title("3.7.1 Univariate Predictive Power")
        plt.tight_layout()
        self._save("3.7.1_predictive_power.png")

    def plot_feature_stability(self, stability_results):
        """3.7.2 Stability (Rolling Correlation Plot)"""
        plt.figure(figsize=(12, 6))

        for feat, data in stability_results.items():
            # data is a Series with MultiIndex (symbol, index) or just index if 1 symbol
            # For visualization, just average across symbols or plot first symbol
            if isinstance(data.index, pd.MultiIndex):
                # Unstack to get mean across symbols per timestamp
                # Reset index to make timestamp accessible
                flat = data.reset_index()
                # Assuming 'level_1' is timestamp or index
                # Ideally, we align by timestamp, but here we plot mean
                avg_corr = flat.groupby(flat.columns[1])[
                    0
                ].mean()  # 0 is the corr value column name usually
                avg_corr.plot(label=feat, alpha=0.7)
            else:
                data.plot(label=feat, alpha=0.7)

        plt.title("3.7.2 Feature Stability (Rolling Correlation)")
        plt.ylabel("Correlation with Target")
        plt.xlabel("Time")
        plt.legend()
        self._save("3.7.2_feature_stability.png")

    def _save(self, filename):
        path = self.plots_dir / filename
        plt.savefig(path, dpi=150)
        self.logger.info(f"Saved plot: {path}")
        plt.close()


# class FeaturePlots:
#     def __init__(self, plots_dir: Path, logger):
#         self.plots_dir = plots_dir
#         self.logger = logger
#         self.plots_dir.mkdir(parents=True, exist_ok=True)
#         sns.set_theme(style="whitegrid")

#     def generate_all_plots(self, analysis_results: dict):
#         self.plot_autocorrelation_heatmap(analysis_results["autocorrelation"])
#         self.plot_feature_correlation(analysis_results["correlation_matrix"])
#         self.plot_pca_variance(analysis_results["pca_analysis"])
#         self.plot_predictive_power(analysis_results["predictive_power"])

#     def plot_autocorrelation_heatmap(self, df_ac):
#         """3.2.1 Return Horizon Correlation Heatmap"""
#         plt.figure(figsize=(10, 6))
#         #         sns.heatmap(df_ac, annot=True, cmap="coolwarm", center=0, vmin=-0.2, vmax=0.2)
#         plt.title("3.2.1 Autocorrelation of Returns by Horizon")
#         plt.ylabel("Lag")
#         plt.xlabel("Return Horizon")
#         self._save("3.2.1_autocorrelation_heatmap.png")

#     def plot_feature_correlation(self, df_corr):
#         """3.6.1 Feature Correlation Heatmap"""
#         plt.figure(figsize=(12, 10))
#         #         mask = np.triu(np.ones_like(df_corr, dtype=bool))
#         sns.heatmap(
#             df_corr, mask=mask, cmap="vlag", center=0, square=True, linewidths=0.5
#         )
#         plt.title("3.6.1 Feature Correlation Matrix")
#         plt.tight_layout()
#         self._save("3.6.1_feature_correlation.png")

#     def plot_pca_variance(self, pca_results):
#         """3.6.2 PCA Scree Plot"""
#         exp_var = pca_results["explained_variance"] * 100
#         cum_var = pca_results["cumulative_variance"] * 100
#         n_components = len(exp_var)

#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Bar chart for individual variance
#         ax1.bar(
#             range(1, n_components + 1), exp_var, alpha=0.6, label="Individual Variance"
#         )
#         ax1.set_xlabel("Principal Component")
#         ax1.set_ylabel("Variance Explained (%)")

#         # Line chart for cumulative variance
#         ax2 = ax1.twinx()
#         ax2.plot(
#             range(1, n_components + 1),
#             cum_var,
#             color="red",
#             marker="o",
#             label="Cumulative Variance",
#         )
#         ax2.set_ylabel("Cumulative Variance (%)")
#         ax2.set_ylim(0, 105)

#         plt.title("3.6.2 PCA Explained Variance (Scree Plot)")
#         self._save("3.6.2_pca_variance.png")

#     def plot_predictive_power(self, df_pred):
#         """3.7.1 Predictive Power Bar Chart"""
#         plt.figure(figsize=(12, 6))
#         # Sort by R-squared
#         df_sorted = df_pred.sort_values("r_squared", ascending=True)

#         # Plot
#         bars = plt.barh(df_sorted["feature"], df_sorted["r_squared"], color="teal")
#         plt.xlabel("R-Squared (Next Bar Prediction)")
#         plt.title("3.7.1 Univariate Predictive Power (R-Squared)")

#         # Add value labels
#         for bar in bars:
#             width = bar.get_width()
#             plt.text(
#                 width,
#                 bar.get_y() + bar.get_height() / 2,
#                 f"{width:.5f}",
#                 ha="left",
#                 va="center",
#             )

#         plt.tight_layout()
#         self._save("3.7.1_predictive_power.png")

#     def _save(self, filename):
#         path = self.plots_dir / filename
#         plt.savefig(path, dpi=150)
#         self.logger.info(f"Saved plot: {path}")
#         plt.close()
