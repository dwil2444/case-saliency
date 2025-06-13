import fire
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

def run_agreement_stats(csv_path: str, output_dir: str, threshold: float = 70.0):
    """
    Performs statistical analysis on top-k agreement scores for each saliency method.

    Args:
        csv_path: Path to CSV file with columns [idx, alg, model, top1_label, top2_label, agreement]
        output_dir: Directory to store plots and results
        threshold: Threshold for null hypothesis (e.g., 70% agreement)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    methods = df['alg'].unique()

    log_lines = []
    for method in methods:
        scores = df[df['alg'] == method]['agreement'].dropna()
        stat, p_value = wilcoxon(scores - threshold, alternative='less')

        mean_agree = scores.mean()
        std_agree = scores.std()
        n = len(scores)

        reject = p_value < 0.05
        conclusion = "REJECT" if reject else "FAIL TO REJECT"

        line = (f"[{method}] N={n} | Mean={mean_agree:.2f} | SD={std_agree:.2f} | "
                f"Wilcoxon p={p_value:.4f} -> {conclusion} H₀ (agreement < {threshold}%)")
        print(line)
        log_lines.append(line)

        # Plot relative frequency histogram with KDE
        plt.figure(figsize=(6, 4))
        sns.histplot(scores, bins=20, kde=True, color='steelblue', stat='probability')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}%')
        plt.title(f"Agreement Distribution for {method}")
        plt.xlabel("Agreement (%)")
        plt.ylabel("Relative Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_agreement_dist.pdf"))
        plt.close()

    # Write summary log
    with open(os.path.join(output_dir, "statistical_results.txt"), "w") as f:
        for line in log_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    fire.Fire(run_agreement_stats)
