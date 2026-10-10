from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
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


def normalize_density(f: Expr, region: Region, constant: Symbol) -> Expr:
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
    return f.subs(constant, c_val)


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
    normalized_f = normalize_density(f, region_x, c)  # x or y is both fine
    print(f"Normalized f(x, y) = {normalized_f}")

    # task 2
    f_X = sp.simplify(marginal(normalized_f, region_x))
    F_X = sp.simplify(sp.integrate(f_X, x))
    print(f"f_X(x) = {f_X}")
    print(f"F_X(x) = {F_X}")
    f_Y = sp.simplify(marginal(normalized_f, region_y))
    F_Y = sp.simplify(sp.integrate(f_Y, y))
    print(f"f_Y(y) = {f_Y}")
    print(f"F_Y(y) = {F_Y}")

    domain: Expr = (y <= x + 2) & (y <= 4 - x) & (x <= 2) & (x >= 0) & (y >= 0)

    visualize_density(normalized_f, domain, x, y)
    visualize_marginals(f_X, F_X, f_Y, F_Y, x, y)


if __name__ == "__main__":
    main()
