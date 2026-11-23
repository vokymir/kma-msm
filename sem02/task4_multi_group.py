"""Task 4 – Multi-sample tests of location (ANOVA + Tukey, Kruskal–Wallis + Dunn)."""

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from config import SPECIES, VARIABLES, ALPHA
from common_stats import all_group_values, add_holm_to_rows


# ---------------------------------------------------------------- ANOVA

# [Task 4 | ANOVA] Fit the one-way model "variable ~ species".
def fit_anova_model(data, variable):
    return ols(f"{variable} ~ C(Species)", data=data).fit()


# [Task 4 | ANOVA] ANOVA table (type II sums of squares).
def anova_table(model):
    return anova_lm(model, typ=2)


# [Task 4 | ANOVA] Effect size eta^2 = SS_between / SS_total.
def eta_squared(table):
    return table.loc["C(Species)", "sum_sq"] / table["sum_sq"].sum()


# [Task 4 | ANOVA assumptions] Levene test of equal variances (median-centred).
def levene_p_value(data, variable):
    return stats.levene(*all_group_values(data, variable), center="median").pvalue


# [Task 4 | ANOVA assumptions] Shapiro–Wilk test of normality of residuals.
def residual_shapiro_p_value(model):
    return stats.shapiro(model.resid).pvalue


# [Task 4 | ANOVA] One results row: F, p, effect size and assumption checks.
def anova_row(data, variable):
    model = fit_anova_model(data, variable)
    table = anova_table(model)
    return {
        "promenna": variable, "F": table.loc["C(Species)", "F"],
        "p": table.loc["C(Species)", "PR(>F)"], "eta_squared": eta_squared(table),
        "shapiro_rezidua_p": residual_shapiro_p_value(model),
        "levene_p": levene_p_value(data, variable),
    }


# ---------------------------------------------------------------- Tukey

# [Task 4 | post-hoc, parametric] Convert one Tukey HSD result line to a dictionary.
def tukey_row(variable, line):
    return {
        "promenna": variable, "skupina_1": line[0], "skupina_2": line[1],
        "rozdil": line[2], "dolni_mez": line[3], "horni_mez": line[4],
        "p_adj": line[5], "zamitame": line[6],
    }


# [Task 4 | post-hoc, parametric] Tukey HSD pairwise comparisons for one variable.
def tukey_rows(data, variable):
    tukey = pairwise_tukeyhsd(data[variable], data.Species, alpha=ALPHA)
    return [tukey_row(variable, line) for line in tukey._results_table.data[1:]]


# ---------------------------------------------------------------- Kruskal–Wallis

# [Task 4 | non-parametric omnibus test] Kruskal–Wallis test for one variable.
def kruskal_row(data, variable):
    kw = stats.kruskal(*all_group_values(data, variable))
    return {"promenna": variable, "H": kw.statistic, "p": kw.pvalue}


# ---------------------------------------------------------------- Dunn

# [Task 4 | post-hoc, non-parametric] Ranks of all observations of one variable.
def rank_observations(data, variable):
    ranked = data[["Species", variable]].copy()
    ranked["rank"] = stats.rankdata(ranked[variable])
    return ranked


# [Task 4 | post-hoc, non-parametric] Correction factor for tied values.
def tie_correction(values):
    total_n = len(values)
    tie_sizes = values.value_counts().to_numpy()
    return 1 - np.sum(tie_sizes**3 - tie_sizes) / (total_n**3 - total_n)


# [Task 4 | post-hoc, non-parametric] Variance of the ranks (with tie correction).
def rank_variance(ranked, variable):
    total_n = len(ranked)
    return total_n * (total_n + 1) / 12 * tie_correction(ranked[variable])


# [Task 4 | post-hoc, non-parametric] Dunn z-test for one pair of groups.
def dunn_pair_row(ranked, variable, variance, first, second):
    first_ranks = ranked.loc[ranked.Species == first, "rank"]
    second_ranks = ranked.loc[ranked.Species == second, "rank"]
    standard_error = np.sqrt(variance * (1 / len(first_ranks) + 1 / len(second_ranks)))
    z_value = (first_ranks.mean() - second_ranks.mean()) / standard_error
    return {"promenna": variable, "skupina_1": first, "skupina_2": second,
            "z": z_value, "p": 2 * stats.norm.sf(abs(z_value))}


# [Task 4 | post-hoc, non-parametric] Dunn tests of all pairs, Holm-adjusted.
def dunn_rows(data, variable):
    ranked = rank_observations(data, variable)
    variance = rank_variance(ranked, variable)
    rows = [dunn_pair_row(ranked, variable, variance, first, second)
            for first, second in combinations(SPECIES, 2)]
    return add_holm_to_rows(rows)


# ---------------------------------------------------------------- entry point

# [Task 4 | main entry] Run all multi-sample tests; return four result tables.
def multiple_group_tests(data):
    anova = [anova_row(data, variable) for variable in VARIABLES]
    kruskal = [kruskal_row(data, variable) for variable in VARIABLES]
    tukey, dunn = [], []
    for variable in VARIABLES:
        tukey.extend(tukey_rows(data, variable))
        dunn.extend(dunn_rows(data, variable))
    return (pd.DataFrame(anova), pd.DataFrame(tukey),
            pd.DataFrame(kruskal), pd.DataFrame(dunn))
