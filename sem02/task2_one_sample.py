"""Task 2 – One-sample tests of location (each group vs. overall mean / median)."""

import pandas as pd
from scipy import stats
from config import SPECIES, VARIABLES
from common_stats import group_values, add_holm_column


# [Task 2 | reference value] Overall mean of all chosen fish = mu_0 for the t-test.
def reference_mean(data, variable):
    return data[variable].mean()


# [Task 2 | reference value] Overall median of all chosen fish = reference for the sign test.
def reference_median(data, variable):
    return data[variable].median()


# [Task 2 | parametric test] One-sample t-test: H0 "group mean = reference mean".
def t_test_one_sample(values, mean_reference):
    return stats.ttest_1samp(values, popmean=mean_reference)


# [Task 2 | non-parametric test] Differences from the reference median, zeros removed.
def nonzero_differences(values, median_reference):
    differences = values - median_reference
    return differences[differences != 0]


# [Task 2 | non-parametric test] Sign test: H0 "group median = reference median".
def sign_test_p_value(values, median_reference):
    nonzero = nonzero_differences(values, median_reference)
    positives = int((nonzero > 0).sum())
    return stats.binomtest(positives, n=len(nonzero), p=0.5).pvalue


# [Task 2 | normality check] Shapiro–Wilk p-value of one group.
def shapiro_p_value(values):
    return stats.shapiro(values).pvalue


# [Task 2 | results table] Build one table row (one variable x one species).
def one_sample_row(values, variable, species, mean_reference, median_reference):
    t_test = t_test_one_sample(values, mean_reference)
    return {
        "promenna": variable, "druh": species, "n": len(values),
        "prumer": values.mean(), "median": values.median(),
        "reference_prumer": mean_reference,
        "rozdil_prumeru": values.mean() - mean_reference,
        "t_stat": t_test.statistic, "p_t": t_test.pvalue,
        "reference_median": median_reference,
        "rozdil_medianu": values.median() - median_reference,
        "p_sign": sign_test_p_value(values, median_reference),
        "n_bez_shod": len(nonzero_differences(values, median_reference)),
        "p_shapiro": shapiro_p_value(values),
    }


# [Task 2 | results table] All rows (all species) for one variable.
def one_sample_rows_for_variable(data, variable):
    mean_ref = reference_mean(data, variable)
    median_ref = reference_median(data, variable)
    return [
        one_sample_row(group_values(data, species, variable), variable,
                       species, mean_ref, median_ref)
        for species in SPECIES
    ]


# [Task 2 | multiple testing] Holm correction of both tests, per variable.
def add_one_sample_holm(table):
    add_holm_column(table, "p_t", "p_t_holm")
    add_holm_column(table, "p_sign", "p_sign_holm")
    return table


# [Task 2 | main entry] Run all one-sample tests and return the results table.
def one_sample_tests(data):
    rows = []
    for variable in VARIABLES:
        rows.extend(one_sample_rows_for_variable(data, variable))
    return add_one_sample_holm(pd.DataFrame(rows))
