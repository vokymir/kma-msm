"""Task 3 – Visualisation: matrix (heatmap) of Holm-adjusted pairwise p-values."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import SPECIES, VARIABLES, LABELS, CZECH_SPECIES
from common_plots import save_figure


# [Task 3 | p-value matrix] Symmetric matrix of adjusted p-values for one variable.
def p_value_matrix(pairwise, variable):
    matrix = np.ones((len(SPECIES), len(SPECIES)))
    subset = pairwise.loc[pairwise.promenna == variable]
    for _, row in subset.iterrows():
        i, j = SPECIES.index(row.skupina_1), SPECIES.index(row.skupina_2)
        matrix[i, j] = matrix[j, i] = row.p_welch_holm
    return matrix


# [Task 3 | p-value matrix] Draw one heatmap panel (one variable).
def draw_p_value_heatmap(ax, pairwise, variable):
    labels = [CZECH_SPECIES[s] for s in SPECIES]
    sns.heatmap(
        p_value_matrix(pairwise, variable),
        annot=True,
        fmt=".3g",
        cmap="crest_r",
        vmin=0,
        vmax=0.05,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Holm-korigovaná p-hodnota"},
    )
    ax.set_title(LABELS[variable])
    ax.set_xlabel("")
    ax.set_ylabel("")


# [Task 3 | p-value matrix] Figure 03: heatmaps of Welch test p-values.
def plot_pairwise_heatmap(pairwise):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for ax, variable in zip(axes, VARIABLES):
        draw_p_value_heatmap(ax, pairwise, variable)
    fig.suptitle("Párová porovnání – Welchův t-test")
    save_figure(fig, "03_parova_porovnani.png")
