import os
import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_classwise_kde(csv_path,
                       model_name='model',
                       output_path='kde_classwise_activity.pdf'):
    sns.set(style="whitegrid")

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'label' not in df or 'active_channel_count' not in df:
        print(f"[ERROR] CSV must contain 'label' and 'active_channel_count'")
        return

    class_means = df.groupby("label")["active_channel_count"].mean().reset_index()
    values = class_means["active_channel_count"].values

    plt.figure(figsize=(8, 4))
    sns.kdeplot(values, fill=True, label=model_name, linewidth=2)
    plt.xlabel("Mean Active Channel Count per Class")
    plt.ylabel("Density")
    plt.title(f"Per-Class Channel Sparsity (KDE) – {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Saved] KDE plot to {output_path}")


if __name__ == '__main__':
    fire.Fire(plot_classwise_kde)
