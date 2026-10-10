from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sympy as sp
from sympy import Expr, Symbol

Limit: TypeAlias = Expr | float | int


@dataclass
class SubRegion:
    """
    Represents: inner_from <= inner_var <= inner_to,
    for outer_from <= outer_var <= outer_to.
    """

    inner_var: Symbol
    inner_from: Limit
    inner_to: Limit
    outer_var: Symbol
    outer_from: Limit
    outer_to: Limit


Region: TypeAlias = list[SubRegion]


@dataclass
class TheoreticalCharacteristics:
    mean_x: Expr
    mean_y: Expr
    var_x: Expr
    var_y: Expr
    cov_xy: Expr
    cov_matrix: sp.Matrix


@dataclass
class EmpiricalCharacteristics:
    mean_x: float
    mean_y: float
    var_x: float
    var_y: float
    cov_xy: float
    cov_matrix: np.ndarray
    ci_mean_x: tuple[float, float]
    ci_mean_y: tuple[float, float]
    ci_var_x: tuple[float, float]
    ci_var_y: tuple[float, float]


@dataclass
class GoodnessOfFitResults:
    ks_stat_x: float
    p_value_x: float
    ks_stat_y: float
    p_value_y: float


def normalize_density(f: Expr, region: Region, constant: Symbol) -> (Expr, Expr):
    """
    Integrate f across all sub-regions and solve for constant c.
    """
    total = sum(
        sp.integrate(
            f,
            (sub.inner_var, sub.inner_from, sub.inner_to),
            (sub.outer_var, sub.outer_from, sub.outer_to),
        )
        for sub in region
    )
    c_val = sp.solve(sp.Eq(total, 1), constant)[0]
    return c_val, f.subs(constant, c_val)


def marginal(f: Expr, region: Region) -> Expr:
    """
    Calculate marginal density by summing inner integrals for matching outer slices.
    """
    # Group integration pieces by outer condition ranges
    pieces = []
    for sub in region:
        piece = sp.integrate(f, (sub.inner_var, sub.inner_from, sub.inner_to))
        cond = (sub.outer_from <= sub.outer_var) & (sub.outer_var <= sub.outer_to)
        pieces.append((piece, cond))

    pieces.append((0, True))
    return sp.Piecewise(*pieces)


def calculate_characteristics(
    f: Expr, region: Region, x: Symbol, y: Symbol
) -> TheoreticalCharacteristics:
    """
    Calculate E[X], E[Y], Var(X), Var(Y), Cov(X, Y), and the covariance matrix.
    """

    def expect(g: Expr) -> Expr:
        """
        Helper to calculate expectation E[g(X, Y)].
        """
        return sum(
            sp.integrate(
                g * f,
                (sub.inner_var, sub.inner_from, sub.inner_to),
                (sub.outer_var, sub.outer_from, sub.outer_to),
            )
            for sub in region
        )

    e_x = expect(x)
    e_y = expect(y)
    e_x2 = expect(x**2)
    e_y2 = expect(y**2)
    e_xy = expect(x * y)

    var_x = e_x2 - e_x**2
    var_y = e_y2 - e_y**2
    cov_xy = e_xy - e_x * e_y

    cov_mat = sp.Matrix([[var_x, cov_xy], [cov_xy, var_y]])

    return TheoreticalCharacteristics(
        mean_x=e_x,
        mean_y=e_y,
        var_x=var_x,
        var_y=var_y,
        cov_xy=cov_xy,
        cov_matrix=cov_mat,
    )


def generate_samples(
    f: Expr,
    domain_condition: Expr,
    x: Symbol,
    y: Symbol,
    n_samples: int = 1000,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0, 2), (0, 3)),
) -> np.ndarray:
    """
    Generate a random sample of (X, Y) vectors from density f(x, y)
    using the Rejection Sampling method.
    """
    f_num = sp.lambdify((x, y), f, modules="numpy")
    cond_num = sp.lambdify((x, y), domain_condition, modules="numpy")

    (x_min, x_max), (y_min, y_max) = bounds

    # Estimate maximum height M of density function f(x, y) over domain
    xg = np.linspace(x_min, x_max, 200)
    yg = np.linspace(y_min, y_max, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    M = np.max(np.where(cond_num(Xg, Yg), f_num(Xg, Yg), 0)) * 1.05

    samples = []
    batch_size = n_samples * 5

    while sum(len(s) for s in samples) < n_samples:
        x_rand = np.random.uniform(x_min, x_max, batch_size)
        y_rand = np.random.uniform(y_min, y_max, batch_size)
        z_rand = np.random.uniform(0, M, batch_size)

        valid_domain = cond_num(x_rand, y_rand)
        accepted = valid_domain & (z_rand <= f_num(x_rand, y_rand))

        if np.any(accepted):
            samples.append(np.column_stack((x_rand[accepted], y_rand[accepted])))

    return np.vstack(samples)[:n_samples]

def calculate_empirical_characteristics(
    samples: np.ndarray, alpha: float = 0.05
) -> EmpiricalCharacteristics:
    """
    Calculate point estimates and confidence intervals (for means & variances) from sample data.
    """
    n = len(samples)
    x_data = samples[:, 0]
    y_data = samples[:, 1]

    # Point Estimates
    mean_x = float(np.mean(x_data))
    mean_y = float(np.mean(y_data))

    var_x = float(np.var(x_data, ddof=1))
    var_y = float(np.var(y_data, ddof=1))

    cov_matrix = np.cov(x_data, y_data, ddof=1)
    cov_xy = float(cov_matrix[0, 1])

    # Confidence Intervals for Means (Student's t-distribution)
    t_crit = st.t.ppf(1 - alpha / 2, df=n - 1)
    se_x = np.sqrt(var_x / n)
    se_y = np.sqrt(var_y / n)

    ci_mean_x = (mean_x - t_crit * se_x, mean_x + t_crit * se_x)
    ci_mean_y = (mean_y - t_crit * se_y, mean_y + t_crit * se_y)

    # Confidence Intervals for Variances (Chi-Square distribution)
    chi2_lower = st.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = st.chi2.ppf(1 - alpha / 2, df=n - 1)

    ci_var_x = ((n - 1) * var_x / chi2_upper, (n - 1) * var_x / chi2_lower)
    ci_var_y = ((n - 1) * var_y / chi2_upper, (n - 1) * var_y / chi2_lower)

    return EmpiricalCharacteristics(
        mean_x=mean_x,
        mean_y=mean_y,
        var_x=var_x,
        var_y=var_y,
        cov_xy=cov_xy,
        cov_matrix=cov_matrix,
        ci_mean_x=ci_mean_x,
        ci_mean_y=ci_mean_y,
        ci_var_x=ci_var_x,
        ci_var_y=ci_var_y,
    )


def calculate_goodness_of_fit(
    samples: np.ndarray, F_X: Expr, F_Y: Expr, x: Symbol, y: Symbol
) -> GoodnessOfFitResults:
    """
    Evaluate agreement between sample data and theoretical CDFs using the Kolmogorov-Smirnov test.
    """
    x_data = samples[:, 0]
    y_data = samples[:, 1]

    F_X_num = sp.lambdify(x, F_X, modules="numpy")
    F_Y_num = sp.lambdify(y, F_Y, modules="numpy")

    ks_x = st.kstest(x_data, F_X_num)
    ks_y = st.kstest(y_data, F_Y_num)

    return GoodnessOfFitResults(
        ks_stat_x=float(ks_x.statistic),
        p_value_x=float(ks_x.pvalue),
        ks_stat_y=float(ks_y.statistic),
        p_value_y=float(ks_y.pvalue),
    )

def visualize_density(f: Expr, domain_condition: Expr, x: Symbol, y: Symbol) -> None:
    """
    Visualize density function on domain.
    """
    f_num = sp.lambdify((x, y), f, modules="numpy")
    cond_num = sp.lambdify((x, y), domain_condition, modules="numpy")

    X, Y = np.meshgrid(
        np.linspace(-3, 3, 300),
        np.linspace(-3, 3, 300),
    )

    # Evaluate the Boolean expression directly across the 2D grid
    mask = cond_num(X, Y)
    Z = np.where(mask, f_num(X, Y), np.nan)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)

    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.set_zlabel("f(x, y)")
    ax.set_title(f"Density function {f}")

    plt.tight_layout()
    plt.show()


def visualize_marginals(
    f_X: Expr, F_X: Expr, f_Y: Expr, F_Y: Expr, x: Symbol, y: Symbol
) -> None:
    """
    Plot 1D marginal for X and Y in a 2x2 grid.
    """
    f_X_num = sp.lambdify(x, f_X, modules="numpy")
    F_X_num = sp.lambdify(x, F_X, modules="numpy")
    f_Y_num = sp.lambdify(y, f_Y, modules="numpy")
    F_Y_num = sp.lambdify(y, F_Y, modules="numpy")

    x_vals = np.linspace(-0.5, 2.5, 300)
    y_vals = np.linspace(-0.5, 3.5, 300)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # X
    axes[0, 0].plot(x_vals, f_X_num(x_vals), color="tab:blue", lw=2)
    axes[0, 0].set_title("$f_X(x)$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].grid(True)

    axes[0, 1].plot(x_vals, F_X_num(x_vals), color="tab:orange", lw=2)
    axes[0, 1].set_title("$F_X(x)$")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True)

    # Y
    axes[1, 0].plot(y_vals, f_Y_num(y_vals), color="tab:green", lw=2)
    axes[1, 0].set_title("$f_Y(y)$")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].grid(True)

    axes[1, 1].plot(y_vals, F_Y_num(y_vals), color="tab:red", lw=2)
    axes[1, 1].set_title("$F_Y(y)$")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_samples(samples: np.ndarray) -> None:
    """
    Plot generated 2D random sample points.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=15, color="tab:purple")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Generated Random Sample (N={len(samples)})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_empirical_vs_theoretical(
    samples: np.ndarray,
    f_X: Expr,
    F_X: Expr,
    f_Y: Expr,
    F_Y: Expr,
    x: Symbol,
    y: Symbol,
    gof: GoodnessOfFitResults | None = None,
) -> None:
    """
    Plot sample histograms and ECDFs overlaid with theoretical marginal PDFs and CDFs.
    """
    x_data = np.sort(samples[:, 0])
    y_data = np.sort(samples[:, 1])
    n = len(samples)

    f_X_num = sp.lambdify(x, f_X, modules="numpy")
    F_X_num = sp.lambdify(x, F_X, modules="numpy")
    f_Y_num = sp.lambdify(y, f_Y, modules="numpy")
    F_Y_num = sp.lambdify(y, F_Y, modules="numpy")

    x_grid = np.linspace(-0.2, 2.2, 300)
    y_grid = np.linspace(-0.2, 3.2, 300)

    ecdf_x = np.arange(1, n + 1) / n
    ecdf_y = np.arange(1, n + 1) / n

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Plot 1: Histogram X vs PDF f_X(x) ---
    axes[0, 0].hist(
        x_data,
        bins=25,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Sample Histogram",
    )
    axes[0, 0].plot(
        x_grid, f_X_num(x_grid), color="darkblue", lw=2.5, label="Theoretical $f_X(x)$"
    )
    axes[0, 0].set_title("Density Comparison (X)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- Plot 2: ECDF X vs CDF F_X(x) ---
    axes[0, 1].step(x_data, ecdf_x, where="post", color="orange", label="Empirical CDF (ECDF)")
    axes[0, 1].plot(
        x_grid, F_X_num(x_grid), color="red", linestyle="--", lw=2, label="Theoretical $F_X(x)$"
    )
    title_x = "CDF Comparison (X)"
    if gof:
        title_x += f"\nKS p-value: {gof.p_value_x:.4f}"
    axes[0, 1].set_title(title_x)
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- Plot 3: Histogram Y vs PDF f_Y(y) ---
    axes[1, 0].hist(
        y_data,
        bins=25,
        density=True,
        alpha=0.6,
        color="lightgreen",
        edgecolor="black",
        label="Sample Histogram",
    )
    axes[1, 0].plot(
        y_grid, f_Y_num(y_grid), color="darkgreen", lw=2.5, label="Theoretical $f_Y(y)$"
    )
    axes[1, 0].set_title("Density Comparison (Y)")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # --- Plot 4: ECDF Y vs CDF F_Y(y) ---
    axes[1, 1].step(y_data, ecdf_y, where="post", color="purple", label="Empirical CDF (ECDF)")
    axes[1, 1].plot(
        y_grid, F_Y_num(y_grid), color="black", linestyle="--", lw=2, label="Theoretical $F_Y(y)$"
    )
    title_y = "CDF Comparison (Y)"
    if gof:
        title_y += f"\nKS p-value: {gof.p_value_y:.4f}"
    axes[1, 1].set_title(title_y)
    axes[1, 1].set_xlabel("y")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()



def main():
    x: Symbol = sp.Symbol("x", real=True)
    y: Symbol = sp.Symbol("y", real=True)
    c: Symbol = sp.Symbol("c", real=True)

    f: Expr = c * x * y**2

    region_x: Region = [
        SubRegion(
            inner_var=y,
            inner_from=0,
            inner_to=x + 2,
            outer_var=x,
            outer_from=0,
            outer_to=1,
        ),
        SubRegion(
            inner_var=y,
            inner_from=0,
            inner_to=4 - x,
            outer_var=x,
            outer_from=1,
            outer_to=2,
        ),
    ]

    region_y: Region = [
        SubRegion(
            inner_var=x, inner_from=0, inner_to=2, outer_var=y, outer_from=0, outer_to=2
        ),
        SubRegion(
            inner_var=x,
            inner_from=y - 2,
            inner_to=4 - y,
            outer_var=y,
            outer_from=2,
            outer_to=3,
        ),
    ]

    # task 1
    c_val, normalized_f = normalize_density(f, region_x, c)  # x or y is both fine
    print(f"c = {c_val}")
    print(f"Normalized f(x, y) = {normalized_f}")
    print()

    # task 2
    f_X = sp.simplify(marginal(normalized_f, region_x))
    F_X = sp.simplify(sp.integrate(f_X, x))
    print(f"f_X(x) = {f_X}")
    print(f"F_X(x) = {F_X}")
    f_Y = sp.simplify(marginal(normalized_f, region_y))
    F_Y = sp.simplify(sp.integrate(f_Y, y))
    print(f"f_Y(y) = {f_Y}")
    print(f"F_Y(y) = {F_Y}")
    print()

    domain: Expr = (y <= x + 2) & (y <= 4 - x) & (x <= 2) & (x >= 0) & (y >= 0)

    # visualize_density(normalized_f, domain, x, y)
    # visualize_marginals(f_X, F_X, f_Y, F_Y, x, y)

    # task 3
    stats = calculate_characteristics(normalized_f, region_x, x, y)
    print(f"E[X]      = {stats.mean_x} (~{float(stats.mean_x):.4f})")
    print(f"E[Y]      = {stats.mean_y} (~{float(stats.mean_y):.4f})")
    print(f"Var(X)    = {stats.var_x} (~{float(stats.var_x):.4f})")
    print(f"Var(Y)    = {stats.var_y} (~{float(stats.var_y):.4f})")
    print(f"Cov(X, Y) = {stats.cov_xy} (~{float(stats.cov_xy):.4f})")
    print("\nCovariance Matrix:")
    sp.pprint(stats.cov_matrix)
    print()

    # task 4
    samples: np.ndarray = generate_samples(normalized_f, domain, x, y,
                                            bounds=((0,2),(0,3)))
    # visualize_samples(samples)

    # task 5
    emp = calculate_empirical_characteristics(samples, alpha=0.05)
    print("=== Sample Estimates & 95% Confidence Intervals ===")
    print(f"Sample Mean X : {emp.mean_x:.4f} | 95% CI: [{emp.ci_mean_x[0]:.4f}, {emp.ci_mean_x[1]:.4f}]")
    print( f"Sample Mean Y : {emp.mean_y:.4f} | 95% CI: [{emp.ci_mean_y[0]:.4f}, {emp.ci_mean_y[1]:.4f}]")
    print( f"Sample Var X  : {emp.var_x:.4f} | 95% CI: [{emp.ci_var_x[0]:.4f}, {emp.ci_var_x[1]:.4f}]")
    print( f"Sample Var Y  : {emp.var_y:.4f} | 95% CI: [{emp.ci_var_y[0]:.4f}, {emp.ci_var_y[1]:.4f}]")
    print(f"Sample Cov XY : {emp.cov_xy:.4f}")
    print("\nSample Covariance Matrix:\n", emp.cov_matrix)

    # task 6
    gof = calculate_goodness_of_fit(samples, F_X, F_Y, x, y)

    print("=== Goodness-of-Fit Evaluation (Kolmogorov-Smirnov Test) ===")
    print(f"X: KS Statistic = {gof.ks_stat_x:.4f}, p-value = {gof.p_value_x:.4f}")
    print(f"Y: KS Statistic = {gof.ks_stat_y:.4f}, p-value = {gof.p_value_y:.4f}")

    visualize_empirical_vs_theoretical(samples, f_X, F_X, f_Y, F_Y, x, y, gof)


if __name__ == "__main__":
    main()
