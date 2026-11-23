"""Task 5 – Visualisation: weight vs. height scatter plot with 95% data ellipses."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from config import SPECIES, VARIABLES, LABELS, CZECH_SPECIES
from common_plots import species_colors, save_figure
import seaborn as sns

PAIR = ["Perch", "Roach"]


# [Task 5 | scatter plot] Points of one species (legend uses the Czech name).
def draw_group_points(ax, group, species, color):
    ax.scatter(
        group.Weight,
        group.Height,
        label=CZECH_SPECIES[species],
        color=color,
        alpha=0.78,
        s=35,
        edgecolor="white",
        linewidth=0.4,
    )


# [Task 5 | data ellipse] Eigen-decomposition of a covariance matrix, largest first.
def sorted_eigen(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


# [Task 5 | data ellipse] Build the 95% data ellipse (Mahalanobis-distance contour).
def data_ellipse(values, color):
    eigenvalues, eigenvectors = sorted_eigen(np.cov(values, rowvar=False))
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(chi2.ppf(0.95, 2) * eigenvalues)
    return Ellipse(
        values.mean(axis=0),
        width,
        height,
        angle=angle,
        fill=False,
        color=color,
        linewidth=2,
    )


# [Task 5 | scatter plot] Points and ellipse of one species.
def draw_group(ax, data, species, color):
    group = data.loc[data.Species == species, VARIABLES]
    draw_group_points(ax, group, species, color)
    ax.add_patch(data_ellipse(group.to_numpy(float), color))


# [Task 5 | scatter plot] Figure 02: all groups with their ellipses.
def plot_scatter_with_ellipses(data):
    colors = species_colors()
    fig, ax = plt.subplots(figsize=(7.1, 5.7))
    for species in SPECIES:
        draw_group(ax, data, species, colors[species])
    ax.set_xlabel(LABELS["Weight"])
    ax.set_ylabel(LABELS["Height"])
    ax.set_title("Vztah hmotnosti a výšky těla")
    # Weight and height cannot be negative: cut the axes at 0. Must come after
    # all patches are added, so the upper limits still include the ellipses.
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title="Druh")
    save_figure(fig, "02_vicerozmerny.png")


# ----- Another image for poster


# [Task 5 | extra figure] Density of one variable for Okoun and Plotice, mean and median marked.
def draw_density(ax, data, variable, colors):
    for species in PAIR:
        values = data.loc[data.Species == species, variable]
        sns.kdeplot(
            values,
            ax=ax,
            color=colors[species],
            fill=True,
            alpha=0.25,
            cut=0,
            linewidth=2,
            label=CZECH_SPECIES[species],
        )
        ax.axvline(values.mean(), color=colors[species], linestyle="--", linewidth=1.5)
        ax.axvline(values.median(), color=colors[species], linestyle=":", linewidth=1.5)
    ax.set_xlabel(LABELS[variable])
    ax.set_ylabel("Hustota")
    ax.set_xlim(left=0)


# [Task 5 | extra figure] Figure 04: Okoun vs. Plotice, weight and height.
def plot_perch_roach_distributions(data):
    colors = species_colors()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for ax, variable in zip(axes, VARIABLES):
        draw_density(ax, data, variable, colors)
    axes[0].legend(title="Druh")
    fig.suptitle("Okoun a plotice: rozdělení znaků")
    save_figure(fig, "06_okoun_plotice.png")
