"""Task 5 – Multivariate approach (MANOVA, Hotelling T^2, Box's M test)."""

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from config import SPECIES, VARIABLES
from common_stats import add_holm_to_rows


# ---------------------------------------------------------------- helpers

# [Task 5 | helper] Observations of one species as an (n x p) float matrix.
def group_matrix(data, species):
    return data.loc[data.Species == species, VARIABLES].to_numpy(float)


# ---------------------------------------------------------------- MANOVA

# [Task 5 | all groups] MANOVA: H0 "mean vectors of all groups are equal".
def manova_test(data):
    formula = " + ".join(VARIABLES) + " ~ C(Species)"
    return MANOVA.from_formula(formula, data=data).mv_test()


# ---------------------------------------------------------------- Hotelling

# [Task 5 | pairwise] Pooled covariance matrix of two groups.
def pooled_covariance_two(x, y):
    n1, n2 = len(x), len(y)
    return ((n1 - 1) * np.cov(x, rowvar=False)
            + (n2 - 1) * np.cov(y, rowvar=False)) / (n1 + n2 - 2)


# [Task 5 | pairwise] Hotelling T^2 statistic for two mean vectors.
def hotelling_t_squared(x, y):
    n1, n2 = len(x), len(y)
    difference = x.mean(axis=0) - y.mean(axis=0)
    covariance = pooled_covariance_two(x, y)
    return n1 * n2 / (n1 + n2) * difference @ np.linalg.solve(covariance, difference)


# [Task 5 | pairwise] Convert T^2 to an F statistic.
def hotelling_f_statistic(t_squared, n1, n2):
    dimension = len(VARIABLES)
    return (n1 + n2 - dimension - 1) * t_squared / (dimension * (n1 + n2 - 2))


# [Task 5 | pairwise] p-value of the F statistic.
def hotelling_p_value(f_statistic, n1, n2):
    dimension = len(VARIABLES)
    return stats.f.sf(f_statistic, dimension, n1 + n2 - dimension - 1)


# [Task 5 | pairwise] One results row: Hotelling test of one pair of groups.
def hotelling_pair_row(data, first, second):
    x, y = group_matrix(data, first), group_matrix(data, second)
    t_squared = hotelling_t_squared(x, y)
    f_statistic = hotelling_f_statistic(t_squared, len(x), len(y))
    return {"skupina_1": first, "skupina_2": second, "T2": t_squared,
            "F": f_statistic, "p": hotelling_p_value(f_statistic, len(x), len(y))}


# [Task 5 | pairwise] Hotelling tests of all pairs, Holm-adjusted.
def hotelling_tests(data):
    rows = [hotelling_pair_row(data, first, second)
            for first, second in combinations(SPECIES, 2)]
    return pd.DataFrame(add_holm_to_rows(rows))


# ---------------------------------------------------------------- Box's M

# [Task 5 | covariance matrices] Log-determinant; fails if matrix is not positive definite.
def log_determinant(matrix):
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Boxův M test nelze spočítat: kovarianční matice není pozitivně definitní.")
    return logdet


# [Task 5 | covariance matrices] Pooled covariance matrix of all groups.
def pooled_covariance_all(covariances, sizes):
    total_n, group_count = int(sizes.sum()), len(sizes)
    weighted = sum((n - 1) * cov for n, cov in zip(sizes, covariances))
    return weighted / (total_n - group_count)


# [Task 5 | covariance matrices] Box's M statistic.
def box_m_statistic(covariances, sizes):
    total_n, group_count = int(sizes.sum()), len(sizes)
    pooled_logdet = log_determinant(pooled_covariance_all(covariances, sizes))
    group_logdets = [log_determinant(cov) for cov in covariances]
    weighted = sum((n - 1) * logdet for n, logdet in zip(sizes, group_logdets))
    return (total_n - group_count) * pooled_logdet - weighted


# [Task 5 | covariance matrices] Correction factor for the chi-square approximation.
def box_m_correction(sizes):
    total_n, group_count, dimension = int(sizes.sum()), len(sizes), len(VARIABLES)
    factor = ((2 * dimension**2 + 3 * dimension - 1)
              / (6 * (dimension + 1) * (group_count - 1)))
    return factor * (np.sum(1 / (sizes - 1)) - 1 / (total_n - group_count))


# [Task 5 | covariance matrices] Box's M test: H0 "covariance matrices are equal".
def box_m_test(data):
    groups = [group_matrix(data, species) for species in SPECIES]
    sizes = np.array([len(group) for group in groups])
    covariances = [np.cov(group, rowvar=False) for group in groups]
    dimension = len(VARIABLES)

    statistic = box_m_statistic(covariances, sizes)
    chi_square = (1 - box_m_correction(sizes)) * statistic
    degrees = (len(groups) - 1) * dimension * (dimension + 1) / 2
    return {"M": statistic, "chi2": chi_square, "df": degrees,
            "p": stats.chi2.sf(chi_square, degrees)}
