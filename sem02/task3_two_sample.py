"""Task 3 – Two-sample tests of location (every pair of groups)."""

from itertools import combinations
import pandas as pd
from scipy import stats
from config import SPECIES, VARIABLES, ALPHA
from common_stats import group_values, add_holm_column


# [Task 3 | parametric test] Welch t-test: H0 "the two group means are equal".
def welch_test(x, y):
    return stats.ttest_ind(x, y, equal_var=False)


# [Task 3 | non-parametric test] Mann–Whitney U test (two-sided).
def mann_whitney_test(x, y):
    return stats.mannwhitneyu(x, y, alternative="two-sided")


# [Task 3 | confidence interval] 95% CI of the mean difference from the Welch test.
def welch_interval(welch):
    return welch.confidence_interval(confidence_level=1 - ALPHA)


# [Task 3 | results table] Build one table row (one variable x one pair).
def pairwise_row(data, variable, first, second):
    x = group_values(data, first, variable)
    y = group_values(data, second, variable)
    welch = welch_test(x, y)
    interval = welch_interval(welch)
    return {
        "promenna": variable, "skupina_1": first, "skupina_2": second,
        "rozdil_prumeru": x.mean() - y.mean(),
        "rozdil_medianu": x.median() - y.median(),
        "dolni_mez_95": interval.low, "horni_mez_95": interval.high,
        "p_welch": welch.pvalue, "p_mw": mann_whitney_test(x, y).pvalue,
    }


# [Task 3 | results table] All pairs of species for one variable.
def pairwise_rows_for_variable(data, variable):
    return [pairwise_row(data, variable, first, second)
            for first, second in combinations(SPECIES, 2)]


# [Task 3 | multiple testing] Holm correction of both tests, per variable.
def add_pairwise_holm(table):
    add_holm_column(table, "p_welch", "p_welch_holm")
    add_holm_column(table, "p_mw", "p_mw_holm")
    return table


# [Task 3 | main entry] Run all pairwise tests and return the results table.
def pairwise_tests(data):
    rows = []
    for variable in VARIABLES:
        rows.extend(pairwise_rows_for_variable(data, variable))
    return add_pairwise_holm(pd.DataFrame(rows))
