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


def visualize(f: Expr, domain_condition: Expr, x: Symbol, y: Symbol) -> None:
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
    ax.set_title(f"Density function {f}\n on {domain_condition}")

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
    visualize(normalized_f, domain, x, y)


if __name__ == "__main__":
    main()
