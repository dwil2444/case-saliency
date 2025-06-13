import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_rel
import fire
from utils.logger import CustomLogger
logger = CustomLogger(__name__).logger

def control_df_sample_size(df: pd.DataFrame, 
                           group_col: str = "alg", 
                           value_col: str = "avg_drop", 
                           random_state: int = 42) -> pd.DataFrame:
    """
    Subsamples each group in the DataFrame to match the smallest group size.

    Args:
        df: Input DataFrame with a column for grouping.
        group_col: Column to group by (default is "alg").
        value_col: Column used for sorting after sampling (optional).
        random_state: Random seed for reproducibility.

    Returns:
        A balanced DataFrame with equal sample size per group.
    """
    min_size = df[group_col].value_counts().min()
    logger.info(f"Subsampling each '{group_col}' group to {min_size} samples for balanced comparison.")

    balanced_df = (
        df.groupby(group_col, group_keys=False)
          .apply(lambda x: x.sample(n=min_size, random_state=random_state).sort_values("idx"))
          .reset_index(drop=True)
    )

    return balanced_df


def analyze(csv_path: str,
            output_dir: str = "results/analysis",
            reference_method: str = "case",
            plot_title: str = "Confidence Drop by Saliency Method",
            save_plots: bool = True):
    """
    Analyze saliency evaluation CSV and generate plots/statistics.

    Args:
        csv_path: Path to saliency_eval.csv
        output_dir: Directory to save plots
        reference_method: Method to compare others against (for t-tests)
        plot_title: Title for plots
        save_plots: If True, saves PDF plots to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Normalize algorithm names to avoid duplicates
    df["alg"] = df["alg"].str.strip()
    df = control_df_sample_size(df)

    # Summary stats
    summary = df.groupby("alg")["avg_drop"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    logger.info("\n--- Average Drop Summary ---")
    logger.info(summary.round(4))

    # Bar plot (matplotlib version for robustness)
    plt.figure(figsize=(10, 6))
    summary_reset = summary.reset_index()
    plt.bar(summary_reset["alg"], summary_reset["mean"], yerr=summary_reset["std"])
    plt.title(plot_title)
    plt.ylabel("Average Drop")
    plt.xlabel("Saliency Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        barplot_path = os.path.join(output_dir, "avg_drop_barplot.pdf")
        plt.savefig(barplot_path)
        logger.info(f"Saved bar plot to: {barplot_path}")
    plt.close()

    # CDF plot
    plt.figure(figsize=(10, 6))
    for alg in df['alg'].unique():
        drops = np.sort(df[df['alg'] == alg]["avg_drop"].values)
        plt.plot(drops, np.linspace(0, 1, len(drops)), label=alg)

    plt.xlabel("Confidence Drop")
    plt.ylabel("Cumulative Proportion")
    plt.title("CDF of Drop Scores")
    plt.legend()
    plt.tight_layout()
    if save_plots:
        cdf_path = os.path.join(output_dir, "drop_cdf.pdf")
        plt.savefig(cdf_path)
        logger.info(f"Saved CDF plot to: {cdf_path}")
    plt.close()

    # Paired t-tests vs. reference_method
    logger.info(f"\n--- Paired t-tests vs. {reference_method} ---")
    ref_df = df[df["alg"] == reference_method].sort_values("idx")
    for method in df["alg"].unique():
        if method == reference_method:
            continue
        test_df = df[df["alg"] == method].sort_values("idx")
        if len(ref_df) != len(test_df):
            logger.info(f"Skipping {method} due to mismatched sample sizes.")
            continue
        stat, p = ttest_rel(ref_df["avg_drop"].values, test_df["avg_drop"].values)
        logger.info(f"{method} vs {reference_method}: p = {p:.5f}")

if __name__ == '__main__':
    fire.Fire(analyze)
