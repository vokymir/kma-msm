"""
KMA/MSM 2026/27 - semestrální práce 1 (zadání na odkazu)
https://github.com/zcu-sediva/MSM/tree/efb1ba00fb83747412e49ea97efb36e23f639565/ulohy/uloha1

javok@students.zcu.cz --- 17. 10. 2026

Program počítá konstantu $c$ pro sdruženou pravděpodobnostní funkci hustoty

$
    f(x,y) = c  x  y^2                (1,3)
$                                (2,0) /\
                                      |  | (2,2)
na oblasti tvaru domečku              |__|
                                  (0,0)  (2,0)
$
    Omega = {(x,y) in R^2 | 0 < x < 2, x+2 < y < 4-x} .
$

Následně vypočítá marginální funkce hustoty a distribuční funkce a teoretické
charakteristiky (střední hodnoty, rozptyl, kovarianci).

Vygeneruje náhodný výběr zamítací metodou, pro něj spočte empirické
charakteristiky (střední hodnoty, rozptyly a koviarianci) s intervalovým
odhadem.

Nakonec spočtením šířky intervalů středních hodnot a rozptylů pro různě velká
$n$ ukáže závislost na rozsahu výběru.

Zjištěné hodnoty vypíše na konzoli a příslušné vygenerované obrázky uloží ve
formátu `.svg` a `.png` do složky `./output/`.

Výstup je česky, kód se drží standardních anglických názvů.

Technická poznámka:
pro nainstalování knihoven použijte příkaz:
    `pip install -r requirements.txt`
případně použijte uv nebo pipx

Upozornění:
s částmi kódu pomáhala umělá inteligence všeho druhu, hlavně s vizualizačními
funkcemi

"""

# /// script
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "sympy",
# ]
# ///

import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import stats

ALPHA = 0.05  # confidence is (1 - ALPHA) = 95 %
N = 1000
SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
SHOW_FIGURES = False  # True = open the figures in windows after the run

np.random.seed(2026)  # fixed seed (the same numbers on every run)

x, y, c = sp.symbols("x y c", real=True)

# corners of the region (only used for drawing its border)
BORDER_X = [0, 2, 2, 1, 0, 0]
BORDER_Y = [0, 0, 2, 3, 2, 0]


# theory
# 1 - 3
# --------------------------------------------------------------------------- #
def integrate_over_region(g):
    """
    Double integral of g(x, y) over the whole region. Border is hardcoded here:
        0 <= x <= 1    0 <= y <= x + 2
        1 <= x <= 2    0 <= y <= 4 - x
    """
    part1 = sp.integrate(g, (y, 0, x + 2), (x, 0, 1))
    part2 = sp.integrate(g, (y, 0, 4 - x), (x, 1, 2))
    return part1 + part2


def theory():
    """
    Compute c, marginals, and means.
    Returns everything in a dictionary.
    """
    # coefficient: the integral of the density over the region must equal 1
    c_val = sp.solve(integrate_over_region(c * x * y**2) - 1, c)[0]
    f = c_val * x * y**2

    # marginal densities
    fx1 = sp.simplify(sp.integrate(f, (y, 0, x + 2)))  # 0 <= x <= 1
    fx2 = sp.simplify(sp.integrate(f, (y, 0, 4 - x)))  # 1 <= x <= 2
    fy1 = sp.simplify(sp.integrate(f, (x, 0, 2)))  # 0 <= y <= 2
    fy2 = sp.simplify(sp.integrate(f, (x, y - 2, 4 - y)))  # 2 <= y <= 3

    # distribution functions F
    t = sp.Symbol("t", real=True)  # integration variable
    Fx1 = sp.integrate(fx1.subs(x, t), (t, 0, x))
    Fx2 = Fx1.subs(x, 1) + sp.integrate(fx2.subs(x, t), (t, 1, x))
    Fy1 = sp.integrate(fy1.subs(y, t), (t, 0, y))
    Fy2 = Fy1.subs(y, 2) + sp.integrate(fy2.subs(y, t), (t, 2, y))

    f_X = sp.Piecewise((0, x < 0), (fx1, x <= 1), (fx2, x <= 2), (0, True))
    F_X = sp.Piecewise((0, x < 0), (Fx1, x <= 1), (Fx2, x <= 2), (1, True))
    f_Y = sp.Piecewise((0, y < 0), (fy1, y <= 2), (fy2, y <= 3), (0, True))
    F_Y = sp.Piecewise((0, y < 0), (Fy1, y <= 2), (Fy2, y <= 3), (1, True))

    # distribution function must be 1 at the right end
    assert Fx2.subs(x, 2) == 1 and Fy2.subs(y, 3) == 1

    # E[<smth>] = integral of <smth> * f over the region
    EX = integrate_over_region(x * f)
    EY = integrate_over_region(y * f)
    VX = integrate_over_region(x**2 * f) - EX**2
    VY = integrate_over_region(y**2 * f) - EY**2
    CXY = integrate_over_region(x * y * f) - EX * EY

    return {
        "c": c_val,
        "f": f,
        "f_X": f_X,
        "F_X": F_X,
        "f_Y": f_Y,
        "F_Y": F_Y,
        "EX": EX,
        "EY": EY,
        "VX": VX,
        "VY": VY,
        "CXY": CXY,
        # numeric versions of the functions (for plotting and sampling)
        "f_num": sp.lambdify((x, y), f, "numpy"),
        "fX_num": sp.lambdify(x, f_X, "numpy"),
        "FX_num": sp.lambdify(x, F_X, "numpy"),
        "fY_num": sp.lambdify(y, f_Y, "numpy"),
        "FY_num": sp.lambdify(y, F_Y, "numpy"),
    }


# sampling, interval estimates
# 4 - 5
# --------------------------------------------------------------------------- #
def generate_sample(n, f_num):
    """
    Generate point in (dominantní funkce) [0,2]x[0,3]x[0,M]
    If x,y OK & z <= f, keep.
    Points [xs,ys] have exactly the density f.
    """
    M = 1.0  # f never exceeds 1, so its ok
    xs = np.array([])
    ys = np.array([])
    while len(xs) < n:
        xr = np.random.uniform(0, 2, 10000)
        yr = np.random.uniform(0, 3, 10000)
        zr = np.random.uniform(0, M, 10000)
        inside = (yr <= xr + 2) & (yr <= 4 - xr)
        keep = inside & (zr <= f_num(xr, yr))
        xs = np.append(xs, xr[keep])
        ys = np.append(ys, yr[keep])
    return xs[:n], ys[:n]


def eps_mean(data, alpha=ALPHA):
    """
    mean - eps < mu < mean + eps  (same epsilon on both sides)
    student dist:
    eps = t * s / sqrt(n)
    """
    n = len(data)
    s = np.std(data, ddof=1)
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    return t_value * s / np.sqrt(n)


def eps_var(data, alpha=ALPHA):
    """
    s^2 - eps < sigma^2 < s^2 + eps.
    approx (not normal data):
    eps = z * s^2 * sqrt(2 / (n - 1))
    """
    n = len(data)
    s2 = np.var(data, ddof=1)
    z_value = stats.norm.ppf(1 - alpha / 2)
    return z_value * s2 * np.sqrt(2 / (n - 1))


def estimate(xs, ys):
    """
    Empirical estimates of theoretical characteristics.
    """
    return {
        "n": len(xs),
        "mean_x": np.mean(xs),
        "eps_mean_x": eps_mean(xs),
        "mean_y": np.mean(ys),
        "eps_mean_y": eps_mean(ys),
        "var_x": np.var(xs, ddof=1),
        "eps_var_x": eps_var(xs),
        "var_y": np.var(ys, ddof=1),
        "eps_var_y": eps_var(ys),
        "cov": np.cov(xs, ys)[0, 1],
    }


# plots :(
# --------------------------------------------------------------------------- #
def save(name):
    """Save the current figure as PNG and SVG into ./output."""
    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"output/{name}.png", dpi=300)
    plt.savefig(f"output/{name}.svg")
    if not SHOW_FIGURES:
        plt.close()


def plot_density(th):
    """Joint density as a colour map, with the region border and the mean."""
    gx, gy = np.meshgrid(np.linspace(0, 2, 300), np.linspace(0, 3, 450))
    inside = gy <= np.minimum(gx + 2, 4 - gx)
    z = np.where(inside, th["f_num"](gx, gy), np.nan)  # NaN = outside the region

    plt.figure(figsize=(7, 6))
    plt.contourf(gx, gy, z, levels=15, cmap="viridis")
    plt.colorbar(label="f(x, y)")
    plt.plot(BORDER_X, BORDER_Y, "k-", linewidth=2, label="hranice oblasti")
    plt.plot(
        float(th["EX"]),
        float(th["EY"]),
        "r*",
        markersize=15,
        label="střední hodnota (E[X], E[Y])",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sdružená hustota f(x, y)")
    plt.legend()
    save("01_density_2d")


def plot_density_3d(th):
    """Joint density as a 3D surface."""
    # the grid follows the region exactly: for every x, y runs from 0 to the upper border
    gx, t = np.meshgrid(np.linspace(0, 2, 101), np.linspace(0, 1, 100))
    gy = t * np.minimum(gx + 2, 4 - gx)
    z = th["f_num"](gx, gy)

    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    surface = ax.plot_surface(gx, gy, z, cmap="viridis")
    plt.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Sdružená hustota f(x, y) ve 3D")
    ax.view_init(
        elev=30, azim=-130
    )  # camera angle, chosen so that the whole shape is visible
    save("02_density_3d")


def plot_marginal(grid, f_vals, F_vals, mean, name, filename):
    """Marginal density and distribution function in one plot (name is 'X' or 'Y')."""
    plt.figure(figsize=(7, 5))
    plt.plot(grid, f_vals, label=f"marginální hustota $f_{name}$")
    plt.plot(grid, F_vals, label=f"marginální distribuční funkce $F_{name}$")
    plt.axvline(mean, color="k", linestyle=":", label=f"střední hodnota E[{name}]")
    plt.xlabel(name.lower())
    plt.title(f"Marginální rozdělení veličiny {name}")
    plt.legend(loc="upper left")
    plt.grid(True)
    save(filename)


def plot_sample(xs, ys, th):
    """Scatter plot of the sample with theoretical and sample mean."""
    plt.figure(figsize=(7, 6))
    plt.scatter(
        xs,
        ys,
        s=10,
        alpha=0.5,
        color="tab:purple",
        label=f"náhodný výběr (n = {len(xs)})",
    )
    plt.plot(BORDER_X, BORDER_Y, "k-", linewidth=2, label="hranice oblasti")
    plt.plot(
        float(th["EX"]),
        float(th["EY"]),
        "r*",
        markersize=15,
        label="teoretická střední hodnota",
    )
    plt.plot(
        np.mean(xs),
        np.mean(ys),
        "bx",
        markersize=10,
        markeredgewidth=3,
        label="výběrový průměr",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Náhodný výběr ze sdruženého rozdělení")
    plt.legend(loc="lower center")
    plt.grid(True)
    save("05_sample")


def plot_histogram(data, grid, f_vals, name, filename):
    """Histogram of the data with the theoretical marginal density."""
    plt.figure(figsize=(7, 5))
    plt.hist(
        data, bins=25, density=True, alpha=0.6, edgecolor="black", label="histogram"
    )
    plt.plot(grid, f_vals, "r-", linewidth=2, label=f"teoretická hustota $f_{name}$")
    plt.xlabel(name.lower())
    plt.ylabel("hustota")
    plt.title(f"Histogram a hustota veličiny {name}")
    plt.legend(loc="upper left")
    plt.grid(True)
    save(filename)


def plot_ecdf(data, grid, F_vals, name, filename):
    """Empirical distribution function of the data with the theoretical one."""
    n = len(data)
    steps = np.arange(1, n + 1) / n  # heights of the empirical CDF: 1/n, 2/n, ..., 1

    plt.figure(figsize=(7, 5))
    plt.step(np.sort(data), steps, where="post", label="empirická distribuční funkce")
    plt.plot(
        grid,
        F_vals,
        "r--",
        linewidth=2,
        label=f"teoretická distribuční funkce $F_{name}$",
    )
    plt.xlabel(name.lower())
    plt.ylabel("pravděpodobnost")
    plt.title(f"Distribuční funkce veličiny {name}")
    plt.legend(loc="upper left")
    plt.grid(True)
    save(filename)


def column(results, key):
    """Collect one entry (e.g. 'mean_x') from the list of results as a numpy array."""
    return np.array([r[key] for r in results])


def errorbars(n, est, eps, true_value, color, name, shift=1.0):
    """Estimates with interval estimates (est +- eps) + the true value as a dashed line."""
    plt.errorbar(
        n * shift,
        est,
        yerr=eps,
        fmt="o-",
        color=color,
        capsize=3,
        label=f"odhad {name} (95% intervalový odhad)",
    )
    plt.axhline(
        float(true_value),
        color=color,
        linestyle="--",
        label=f"teoretická hodnota {name}",
    )


def finish_size_plot(title, ylabel, filename, legend_columns=1):
    """Common settings of the plots that show the dependence on the sample size."""
    plt.xscale("log")
    plt.xlabel("rozsah výběru n")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=legend_columns, fontsize=8 if legend_columns > 1 else None)
    plt.grid(True)
    save(filename)


def plot_size_means(results, th):
    n = column(results, "n")
    plt.figure(figsize=(7, 5))
    errorbars(
        n,
        column(results, "mean_x"),
        column(results, "eps_mean_x"),
        th["EX"],
        "tab:blue",
        "E[X]",
    )
    errorbars(
        n,
        column(results, "mean_y"),
        column(results, "eps_mean_y"),
        th["EY"],
        "tab:orange",
        "E[Y]",
    )
    finish_size_plot(
        "Odhady středních hodnot podle rozsahu výběru", "hodnota", "10_size_means"
    )


def plot_size_widths(results):
    n = column(results, "n")
    width_x = 2 * column(results, "eps_mean_x")  # whole width of the interval = 2 * eps
    width_y = 2 * column(results, "eps_mean_y")

    plt.figure(figsize=(7, 5))
    plt.plot(n, width_x, "o-", label="šířka intervalového odhadu E[X]")
    plt.plot(n, width_y, "s-", label="šířka intervalového odhadu E[Y]")
    plt.plot(
        n,
        width_x[0] * np.sqrt(n[0] / n),
        "k:",
        label=r"křivka $1/\sqrt{n}$ (pro srovnání)",
    )
    plt.yscale("log")
    finish_size_plot(
        "Šířka intervalového odhadu střední hodnoty",
        "šířka intervalu",
        "11_size_widths",
    )


def plot_size_variances(results, th):
    n = column(results, "n")
    plt.figure(figsize=(7, 5))
    errorbars(
        n,
        column(results, "var_x"),
        column(results, "eps_var_x"),
        th["VX"],
        "tab:green",
        "Var(X)",
    )
    # Y is shifted slightly to the right so the bars do not overlap
    errorbars(
        n,
        column(results, "var_y"),
        column(results, "eps_var_y"),
        th["VY"],
        "tab:red",
        "Var(Y)",
        shift=1.06,
    )
    finish_size_plot(
        "Odhady rozptylů podle rozsahu výběru", "hodnota", "12_size_variances"
    )


def plot_size_covariance(results, th):
    n = column(results, "n")
    plt.figure(figsize=(7, 5))
    plt.plot(
        n, column(results, "cov"), "d-", color="tab:purple", label="odhad Cov(X, Y)"
    )
    plt.axhline(
        float(th["CXY"]),
        color="tab:purple",
        linestyle="--",
        label="teoretická hodnota Cov(X, Y)",
    )
    finish_size_plot(
        "Odhad kovariance podle rozsahu výběru", "hodnota", "13_size_covariance"
    )


def plot_sample_intervals(xs, ys, th):
    """
    Sample scatter plot with the interval estimates:
      - variances: bands around  mean +- s  (s = sample standard deviation),
      - means: zoomed detail in the corner (the interval for a mean is much shorter
        than the spread of the data, so it would be invisible at full scale).
    """
    e = estimate(xs, ys)
    mx, my = e["mean_x"], e["mean_y"]
    ex, ey = e["eps_mean_x"], e["eps_mean_y"]
    true_x, true_y = float(th["EX"]), float(th["EY"])

    # interval estimate of the variance -> interval for the standard deviation (square root of both ends)
    sx_low, sx_high = (
        np.sqrt(e["var_x"] - e["eps_var_x"]),
        np.sqrt(e["var_x"] + e["eps_var_x"]),
    )
    sy_low, sy_high = (
        np.sqrt(e["var_y"] - e["eps_var_y"]),
        np.sqrt(e["var_y"] + e["eps_var_y"]),
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(
        xs,
        ys,
        s=10,
        alpha=0.3,
        color="tab:purple",
        label=f"náhodný výběr (n = {len(xs)})",
    )
    plt.plot(BORDER_X, BORDER_Y, "k-", linewidth=2)

    # bands: where the lines  mean +- s  can lie according to the interval estimate of the variance
    plt.axvspan(
        mx - sx_high,
        mx - sx_low,
        color="tab:green",
        alpha=0.3,
        label="průměr X ± s (pásmo = 95% intervalový odhad rozptylu X)",
    )
    plt.axvspan(mx + sx_low, mx + sx_high, color="tab:green", alpha=0.3)
    plt.axhspan(
        my - sy_high,
        my - sy_low,
        color="tab:red",
        alpha=0.3,
        label="průměr Y ± s (pásmo = 95% intervalový odhad rozptylu Y)",
    )
    plt.axhspan(my + sy_low, my + sy_high, color="tab:red", alpha=0.3)

    plt.plot(true_x, true_y, "r*", markersize=12, label="teoretická střední hodnota")
    plt.plot(
        mx,
        my,
        "bo",
        markersize=5,
        label="výběrový průměr (v detailu s 95% intervalovým odhadem)",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Náhodný výběr s intervalovými odhady")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)

    # detail: small plot in the lower-left corner (position given as fractions of the main plot)
    detail = plt.gca().inset_axes([0.11, 0.10, 0.30, 0.24])
    detail.errorbar(
        mx,
        my,
        xerr=ex,
        yerr=ey,
        fmt="o",
        color="blue",
        capsize=4,
        elinewidth=1.5,
        capthick=1.5,
    )
    detail.plot(true_x, true_y, "r*", markersize=12)
    # zoom so that both the interval and the theoretical value are visible
    x_low, x_high = min(mx - ex, true_x), max(mx + ex, true_x)
    y_low, y_high = min(my - ey, true_y), max(my + ey, true_y)
    detail.set_xlim(x_low - 0.3 * (x_high - x_low), x_high + 0.3 * (x_high - x_low))
    detail.set_ylim(y_low - 0.3 * (y_high - y_low), y_high + 0.3 * (y_high - y_low))
    detail.tick_params(labelsize=7)
    detail.set_title("detail: průměr ± interv. odhad", fontsize=8)
    detail.grid(True)
    save("14_sample_intervals")


def plot_hist_and_ecdf(data, grid, f_vals, F_vals, name, filename):
    """Histogram + theoretical density + empirical CDF + theoretical CDF in one plot."""
    n = len(data)
    steps = np.arange(1, n + 1) / n  # heights of the empirical CDF: 1/n, 2/n, ..., 1

    plt.figure(figsize=(7, 5))
    plt.hist(
        data,
        bins=25,
        density=True,
        alpha=0.4,
        color="tab:blue",
        edgecolor="black",
        label="histogram",
    )
    plt.plot(
        grid,
        f_vals,
        color="tab:blue",
        linewidth=2.5,
        label=f"teoretická hustota $f_{name}$",
    )
    plt.step(
        np.sort(data),
        steps,
        where="post",
        color="tab:orange",
        linewidth=2,
        label="empirická distribuční funkce",
    )
    plt.plot(
        grid,
        F_vals,
        "--",
        color="tab:red",
        linewidth=2,
        label=f"teoretická distribuční funkce $F_{name}$",
    )
    plt.ylim(0, 1.3)  # free space at the top for the legend
    plt.xlabel(name.lower())
    plt.ylabel("hustota / distribuční funkce")
    plt.title(f"Veličina {name}: histogram a empirická distribuční funkce")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.grid(True)
    save(filename)


def plot_size_estimates_all(results, th):
    """
    Means, variances and covariance in one plot. Every estimate is divided by its
    true value, so all of them can share one axis: 1 = exact agreement with the theory.
    """
    n = column(results, "n")

    def ratio_bars(est_key, eps_key, true_value, color, marker, name, shift):
        # shift: small multiple of n so the bars of different quantities do not overlap
        true_value = float(true_value)
        plt.errorbar(
            n * shift,
            column(results, est_key) / true_value,
            yerr=column(results, eps_key) / true_value,
            fmt=marker + "-",
            color=color,
            linewidth=1,
            capsize=2,
            label=name,
        )

    plt.figure(figsize=(7, 5))
    plt.axhline(1, color="k", linestyle="--", label="teoretická hodnota (= 1)")
    ratio_bars("mean_x", "eps_mean_x", th["EX"], "tab:blue", "o", "E[X]", 0.80)
    ratio_bars("mean_y", "eps_mean_y", th["EY"], "tab:orange", "s", "E[Y]", 0.90)
    ratio_bars("var_x", "eps_var_x", th["VX"], "tab:green", "^", "Var(X)", 1.00)
    ratio_bars("var_y", "eps_var_y", th["VY"], "tab:red", "v", "Var(Y)", 1.11)
    plt.plot(
        n * 1.23,
        column(results, "cov") / float(th["CXY"]),
        "d-",
        color="tab:purple",
        linewidth=1,
        label="Cov(X, Y) (bez intervalu)",
    )
    plt.ylim(0.55, 2.0)  # free space at the top for the legend
    finish_size_plot(
        "Odhady podle rozsahu výběru (odhad / teoretická hodnota)",
        "odhad / teoretická hodnota",
        "17_size_estimates_all",
        legend_columns=3,
    )


def plot_size_widths_all(results):
    """Width of the interval estimates of means and variances in one plot."""
    n = column(results, "n")
    plt.figure(figsize=(7, 5))
    for key, name, marker in [
        ("eps_mean_x", "E[X]", "o"),
        ("eps_mean_y", "E[Y]", "s"),
        ("eps_var_x", "Var(X)", "^"),
        ("eps_var_y", "Var(Y)", "v"),
    ]:
        plt.plot(n, 2 * column(results, key), marker + "-", label=name)
    width_0 = 2 * results[0]["eps_mean_x"]
    plt.plot(
        n,
        width_0 * np.sqrt(n[0] / n),
        "k:",
        label=r"křivka $1/\sqrt{n}$ (pro srovnání)",
    )
    plt.yscale("log")
    finish_size_plot(
        "Šířka intervalových odhadů podle rozsahu výběru",
        "šířka intervalového odhadu",
        "18_size_widths_all",
    )


# main
# --------------------------------------------------------------------------- #
def print_interval(label, estimate_value, eps):
    low, high = estimate_value - eps, estimate_value + eps
    print(
        f"{label:<20}: {estimate_value:.4f} ± {eps:.4f}   intervalový odhad ({low:.4f}, {high:.4f})"
    )


def main():
    # 1 - 3
    th = theory()
    print(f"konstanta c = {th['c']}")
    print(f"marginální hustota f_X(x) = {th['f_X']}")
    print(f"marginální distribuční funkce F_X(x) = {th['F_X']}")
    print(f"marginální hustota f_Y(y) = {th['f_Y']}")
    print(f"marginální distribuční funkce F_Y(y) = {th['F_Y']}")
    print()
    for label, key in [
        ("střední hodnota E[X]", "EX"),
        ("střední hodnota E[Y]", "EY"),
        ("rozptyl Var(X)", "VX"),
        ("rozptyl Var(Y)", "VY"),
        ("kovariance Cov(X, Y)", "CXY"),
    ]:
        print(f"{label:<22} = {th[key]}  (~ {float(th[key]):.4f})")
    print("Kovarianční (variační) matice:")
    sp.pprint(sp.Matrix([[th["VX"], th["CXY"]], [th["CXY"], th["VY"]]]))
    print()

    plot_density(th)
    plot_density_3d(th)
    xg = np.linspace(-0.2, 2.2, 400)
    yg = np.linspace(-0.2, 3.2, 400)
    plot_marginal(
        xg, th["fX_num"](xg), th["FX_num"](xg), float(th["EX"]), "X", "03_marginal_x"
    )
    plot_marginal(
        yg, th["fY_num"](yg), th["FY_num"](yg), float(th["EY"]), "Y", "04_marginal_y"
    )

    # 4
    xs_all, ys_all = generate_sample(max(SIZES), th["f_num"])
    xs, ys = xs_all[:N], ys_all[:N]
    plot_sample(xs, ys, th)
    plot_sample_intervals(xs, ys, th)

    # 5
    e = estimate(xs, ys)
    print(f"Odhady z výběru (rozsah n = {e['n']}), 95% intervalové odhady:")
    print_interval("střední hodnota X", e["mean_x"], e["eps_mean_x"])
    print_interval("střední hodnota Y", e["mean_y"], e["eps_mean_y"])
    print_interval("rozptyl X", e["var_x"], e["eps_var_x"])
    print_interval("rozptyl Y", e["var_y"], e["eps_var_y"])
    print(f"{'kovariance Cov(X, Y)':<20}: {e['cov']:.4f}")
    print()

    # 6
    plot_histogram(xs, xg, th["fX_num"](xg), "X", "06_histogram_x")
    plot_ecdf(xs, xg, th["FX_num"](xg), "X", "07_ecdf_x")
    plot_histogram(ys, yg, th["fY_num"](yg), "Y", "08_histogram_y")
    plot_ecdf(ys, yg, th["FY_num"](yg), "Y", "09_ecdf_y")
    plot_hist_and_ecdf(
        xs, xg, th["fX_num"](xg), th["FX_num"](xg), "X", "15_hist_ecdf_x"
    )
    plot_hist_and_ecdf(
        ys, yg, th["fY_num"](yg), th["FY_num"](yg), "Y", "16_hist_ecdf_y"
    )

    # 7
    results = [estimate(xs_all[:n], ys_all[:n]) for n in SIZES]
    print("Závislost na rozsahu výběru:")
    print(
        f"{'n':>5} | {'průměr X':>8} | {'šířka X':>8} | {'průměr Y':>8} | {'šířka Y':>8}"
    )
    for r in results:
        print(
            f"{r['n']:>5} | {r['mean_x']:>8.4f} | {2 * r['eps_mean_x']:>8.4f} | "
            f"{r['mean_y']:>8.4f} | {2 * r['eps_mean_y']:>8.4f}"
        )
    plot_size_means(results, th)
    plot_size_widths(results)
    plot_size_variances(results, th)
    plot_size_covariance(results, th)
    plot_size_estimates_all(results, th)
    plot_size_widths_all(results)

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
