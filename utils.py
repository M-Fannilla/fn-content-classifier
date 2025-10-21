import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_label_dist(df: pd.DataFrame, name: str = None) -> None:
    labels = df.columns
    counts = df[labels].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, max(4, len(labels) * 0.3)))
    counts.plot(kind='bar')

    score = (counts.values - np.roll(counts.values, -1))[:-1].mean()

    title = f'Label Distribution (counts) :: SCORE: {score}'
    if name:
        title += f' :: {name}'

    plt.title(title)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.close()
    return counts