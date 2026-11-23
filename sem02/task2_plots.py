"""Task 2 – Visualisation: distributions with reference lines, normality (Q–Q) plots."""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import SPECIES, VARIABLES, LABELS, CZECH_SPECIES
from common_plots import species_colors, save_figure


# [Task 2 | visualisation] Boxplot + individual points of one variable.
def draw_boxplot_with_points(ax, data, variable, colors):
    sns.boxplot(
        data=data,
        x="Species",
        y=variable,
        order=SPECIES,
        ax=ax,
        color="#DCE6ED",
        showfliers=False,
    )
    sns.stripplot(
        data=data,
        x="Species",
        y=variable,
        order=SPECIES,
        ax=ax,
        hue="Species",
        palette=colors,
        legend=False,
        alpha=0.7,
        size=3.5,
        jitter=0.16,
    )
    ax.set_xticks(range(len(SPECIES)))
    ax.set_xticklabels([CZECH_SPECIES[s] for s in SPECIES])


# [Task 2 | visualisation] Dashed overall-mean and dotted overall-median reference lines.
def draw_reference_lines(ax, data, variable):
    ax.axhline(
        data[variable].mean(),
        color="#B24C3D",
        linestyle="--",
        linewidth=1.4,
        label="Celkový průměr",
    )
    ax.axhline(
        data[variable].median(),
        color="#277A78",
        linestyle=":",
        linewidth=1.5,
        label="Celkový medián",
    )


# [Task 2 | visualisation] Axis labels, title and legend of one panel.
def label_distribution_panel(ax, variable):
    ax.set_xlabel("Druh ryby")
    ax.set_ylabel(LABELS[variable])
    ax.set_title(LABELS[variable])
    ax.legend(fontsize=8)


# [Task 2 | visualisation] Draw one complete panel (one variable).
def draw_distribution_panel(ax, data, variable, colors):
    draw_boxplot_with_points(ax, data, variable, colors)
    draw_reference_lines(ax, data, variable)
    label_distribution_panel(ax, variable)


# [Task 2 | visualisation] Figure 01: distributions by species + reference lines.
def plot_distribution(data):
    colors = species_colors()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax, variable in zip(axes, VARIABLES):
        draw_distribution_panel(ax, data, variable, colors)
    fig.suptitle("Rozdělení hmotnosti a výšky podle druhu")
    save_figure(fig, "01_rozdeleni.png")


# [Task 2 | normality check] One Q–Q plot (one species, one variable).
def draw_group_qq(ax, data, species, variable):
    values = data.loc[data.Species == species, variable]
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f"{species} – {LABELS[variable]}")


# [Task 2 | normality check] Figure 04: Q–Q plots of every group and variable.
def plot_group_normality(data):
    fig, axes = plt.subplots(len(SPECIES), len(VARIABLES), figsize=(9, 9))
    for i, species in enumerate(SPECIES):
        for j, variable in enumerate(VARIABLES):
            draw_group_qq(axes[i, j], data, species, variable)
    fig.suptitle("Q–Q grafy: normalita v jednotlivých skupinách")
    save_figure(fig, "04_normalita.png", dpi=220)
