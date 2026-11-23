"""Task 4 – Visualisation: ANOVA residual diagnostics (normality, equal variances)."""

import matplotlib.pyplot as plt
from scipy import stats
from config import VARIABLES, LABELS
from common_plots import save_figure
from task4_multi_group import fit_anova_model


# [Task 4 | residual diagnostics] Q–Q plot of the residuals (normality check).
def draw_residual_qq(ax, model, variable):
    stats.probplot(model.resid, dist="norm", plot=ax)
    ax.set_title(f"Q–Q reziduí – {LABELS[variable]}")


# [Task 4 | residual diagnostics] Residuals vs. fitted values (equal-variance check).
def draw_residuals_vs_fitted(ax, model, variable):
    ax.scatter(model.fittedvalues, model.resid, alpha=0.65, s=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predikované hodnoty")
    ax.set_ylabel("Rezidua")
    ax.set_title(f"Rezidua vs. predikce – {LABELS[variable]}")


# [Task 4 | residual diagnostics] Figure 05: both diagnostics for each variable.
def plot_anova_diagnostics(data):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for j, variable in enumerate(VARIABLES):
        model = fit_anova_model(data, variable)
        draw_residual_qq(axes[0, j], model, variable)
        draw_residuals_vs_fitted(axes[1, j], model, variable)
    save_figure(fig, "05_diagnostika_anova.png", dpi=220)
