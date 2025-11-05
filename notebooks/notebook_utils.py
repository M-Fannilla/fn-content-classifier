import matplotlib.pyplot as plt
import pandas as pd


def plot_label_dist(label_dataframe: pd.DataFrame, name: str = ''):
    label_counts = label_dataframe.sum(axis=0).sort_values(ascending=False)
    total = label_counts.sum()

    plt.figure(figsize=(14, 6))
    bars = plt.bar(label_counts.index.astype(str), label_counts.values, color='#1f77b4')
    plt.xticks(rotation=75, ha='right')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(f'Label Distribution {name} ({int(total):,} total samples | total_rows = {label_dataframe.shape[0]:,})')
    plt.tight_layout()

    # ✅ Annotate bars with percentage labels
    for bar, count in zip(bars, label_counts.values):
        height = bar.get_height()
        pct = count / total * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

    plt.show()
    return label_counts