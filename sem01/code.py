from typing import TypeAlias, cast, NamedTuple

from dataclasses import dataclass
import sympy as sp
from sympy import Expr, Symbol

# typing is better
Limit: TypeAlias = int | Expr
Cond: TypeAlias = sp.logic.boolalg.Boolean


@dataclass
class IntegrationBound:
    symbol_: Symbol
    from_: Limit
    to_: Limit


@dataclass
class BoundedRegion:
    cond_: Cond
    bound_: tuple[IntegrationBound, ...]


Region: TypeAlias = tuple[BoundedRegion, ...]


@dataclass
class BothAxisRegion:
    x: Region
    y: Region


def normalize_density(f: Expr, region: Region, constant: Symbol) -> list[Expr]:
    """
    calculate the value of constant in the density function f
    """
    total: Expr = 0
    for bounded_reg in region:
        tmp: Expr = f
        for bound in bounded_reg.bound_:
            tmp = sp.integrate(tmp, (bound.symbol_, bound.from_, bound.to_))

        total = sp.Add(total, tmp)

    solutions = sp.solve(sp.Eq(total, 1), constant)
    if len(solutions) != 1:
        raise ValueError(f"Expected one normalization constant, got {solutions}")

    return cast(list[Expr], solutions)


def marginal(density: Expr, region: Region) -> Expr:
    """Return the marginal density of the axis variable."""
    # this is the axis/variable whose
    # marginal we are taking, ie. marginalX would have x here
    ax: Symbol = region[0].bound_[-1].symbol_

    pieces = []

    for bounded_reg in region:
        bounds = [
            (b.symbol_, b.from_, b.to_) for b in bounded_reg.bound_ if b.symbol_ != ax
        ]

        piece = sp.integrate(density, *bounds)
        pieces.append((piece, bounded_reg.cond_))

    pieces.append((sp.S.Zero, True))

    return cast(Expr, sp.Piecewise(*pieces))


def main():
    x: Symbol = sp.Symbol("x", real=True)
    y: Symbol = sp.Symbol("y", real=True)
    c: Symbol = sp.Symbol("c", real=True)

    # setting density function f and general region
    f: Expr = c * x * y**2
    print(f"f = {f}")

    # {(x,y): 0<x<2, 0<y<min(x+2,4-x)}
    print("{(x,y): 0<x<2, 0<y<min(x+2,4-x)}")
    region: BothAxisRegion = BothAxisRegion(
        x=(
            BoundedRegion(
                cond_=(0 <= x) & (x <= 1),
                bound_=(
                    IntegrationBound(symbol_=y, from_=0, to_=x + 2),
                    IntegrationBound(symbol_=x, from_=0, to_=1),
                ),
            ),
            BoundedRegion(
                cond_=(1 <= x) & (x <= 2),
                bound_=(
                    IntegrationBound(symbol_=y, from_=0, to_=4 - x),
                    IntegrationBound(symbol_=x, from_=1, to_=2),
                ),
            ),
        ),
        y=(
            BoundedRegion(
                cond_=(0 <= y) & (y <= 2),
                bound_=(
                    IntegrationBound(symbol_=x, from_=0, to_=2),
                    IntegrationBound(symbol_=y, from_=0, to_=2),
                ),
            ),
            BoundedRegion(
                cond_=(2 <= y) & (y <= 3),
                bound_=(
                    IntegrationBound(symbol_=x, from_=y - 2, to_=4 - y),
                    IntegrationBound(symbol_=x, from_=2, to_=3),
                ),
            ),
        ),
    )

    # task 01
    c_value: Expr = normalize_density(f, region.x, c)[0]
    normalized = f.subs(c, c_value)
    print(f"c = {c_value}")
    print(f"f = {normalized}")

    # task 02
    marginal_X = marginal(normalized, region.x)
    marginal_Y = marginal(normalized, region.y)
    print(f"F_X(x) = {marginal_X}\nF_Y(y) = {marginal_Y}")


if __name__ == "__main__":
    main()
