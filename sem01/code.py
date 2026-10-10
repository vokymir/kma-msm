from typing import TypeAlias, cast

import sympy as sp
from sympy import Expr, Symbol

# typing is better
Limit: TypeAlias = int | Expr
Bound: TypeAlias = tuple[Symbol, Limit, Limit]
SubRegion: TypeAlias = tuple[Bound, ...]
Region: TypeAlias = list[SubRegion]


# calculate the value of constant in the density function f
def normalize_density(f: Expr, region: Region, constant: Symbol) -> list[Expr]:
    total: Expr = sp.Add(*(cast(Expr, sp.integrate(f, *subregion))
                           for subregion in region))
    solutions = sp.solve(sp.Eq(total, 1), constant)

    return cast(list[Expr], solutions)


def main():
    x: Symbol = sp.Symbol("x", real=True)
    y: Symbol = sp.Symbol("y", real=True)
    c: Symbol = sp.Symbol("c", real=True)

    # setting density function f and general region
    f: Expr = c * x * y**2

    # {(x,y): 0<x<2, 0<y<min(x+2,4-x)}
    region: Region = [
        ( # left
            (y, 0, x + 2),
            (x, 0, 1)
        ),
        ( # right
            (y, 0, 4 - x),
            (x, 1, 2)
        ),
    ]

    c_value: Expr = normalize_density(f, region, c)[0]
    f = f.subs(c, c_value)
    print(f)


if __name__ == "__main__":
    main()
