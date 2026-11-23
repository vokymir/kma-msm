"""Small statistical helpers shared by several assignment tasks."""

from statsmodels.stats.multitest import multipletests
from config import SPECIES, VARIABLES


# [Shared: tasks 2, 3, 4, 5] Return one group's values of one variable.
def group_values(data, species, variable):
    return data.loc[data.Species == species, variable]


# [Shared: task 4] Return the values of every group as a list (order = SPECIES).
def all_group_values(data, variable):
    return [group_values(data, species, variable) for species in SPECIES]


# [Shared: tasks 2, 3, 4, 5] Holm correction of a list of p-values.
def holm_adjust(p_values):
    return multipletests(p_values, method="holm")[1]


# [Shared: tasks 2, 3] Add a Holm-adjusted column, separately for each variable.
def add_holm_column(table, source_column, target_column):
    for variable in VARIABLES:
        selected = table.promenna == variable
        table.loc[selected, target_column] = holm_adjust(table.loc[selected, source_column])


# [Shared: tasks 4, 5] Add a "p_holm" key to a list of row dictionaries that have key "p".
def add_holm_to_rows(rows):
    adjusted = holm_adjust([row["p"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm"] = p_adjusted
    return rows
