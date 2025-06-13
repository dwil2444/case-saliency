import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fire


def parse_stat_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(
                r"\[(.*?)\] N=(\d+) \| Mean=([\d.]+) \| SD=([\d.]+) \| Wilcoxon p=([\d.]+) -> (.*?) H₀",
                line.strip()
            )
            if match:
                method, N, mean, sd, pval, decision = match.groups()
                data.append({
                    "method": method,
                    "N": int(N),
                    "mean": float(mean),
                    "sd": float(sd),
                    "p_value": float(pval),
                    "decision": decision.strip()
                })
    return pd.DataFrame(data)

def aggregate_and_plot_stats(base_dir):
    all_data = []
    run_dirs = sorted([
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ])

    for run_dir in run_dirs:
        stat_file = os.path.join(run_dir, "statistical_results.txt")
        if os.path.exists(stat_file):
            df = parse_stat_file(stat_file)
            df['run'] = os.path.basename(run_dir)
            all_data.append(df)

    if not all_data:
        print("No statistical_results.txt files found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Save aggregated CSV
    output_csv = os.path.join(base_dir, "aggregated_rq1_stats.csv")
    full_df.to_csv(output_csv, index=False)
    print(f"[✓] Aggregated results saved to {output_csv}")

    # Rejection count per method
    reject_counts = (
        full_df[full_df['decision'] == 'REJECT']
        .groupby('method')
        .size()
        .reset_index(name='rejection_count')
    )

    # Plot rejection counts
    plt.figure(figsize=(8, 4))
    sns.barplot(data=reject_counts, x='method', y='rejection_count', palette='deep')
    plt.title('Number of Runs Rejecting H₀ (agreement < threshold)')
    plt.xlabel('Saliency Method')
    plt.ylabel('Rejection Count')
    plt.ylim(0, full_df['run'].nunique())
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "rejection_counts.pdf"))
    plt.close()

    # Plot mean agreement distribution
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=full_df, x='method', y='mean', palette='pastel')
    plt.title('Distribution of Mean Agreement Across Runs')
    plt.xlabel('Saliency Method')
    plt.ylabel('Mean Agreement (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "mean_agreement_distribution.pdf"))
    plt.close()

    print(f"[✓] Plots saved to {base_dir}")

if __name__ == "__main__":
    fire.Fire(aggregate_and_plot_stats)
