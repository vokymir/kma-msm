"""MSM – úloha 2: Testy o poloze u tří druhů ryb.
Autor: Jakub (V)okoun
Datum: 23. 11. 2026

Porovnání hmotnosti a výšky těla okouna, cejna a plotice.
Výsledky se ukládají do adresáře vystupy/.

Metody odpovídají přednáškám P06 a P07 (t-test, znaménkový test, Welch,
Mann–Whitney, ANOVA + Tukey, Kruskal–Wallis + Dunn, Hotelling T², MANOVA,
Boxův M-test) a doprovodným notebookům.

Použití AI:
    AI byla použita jako pomůcka při strukturování kódu a kontrole syntaxe.
    Volba metod, interpretace a obhajoba výsledků jsou vlastní.

Struktura kódu odpovídá jednotlivým úkolům:
    task1_data.py          – volba a čištění dat
    task2_one_sample.py    (+ task2_plots.py)
    task3_two_sample.py    (+ task3_plots.py)
    task4_multi_group.py   (+ task4_plots.py)
    task5_multivariate.py  (+ task5_plots.py)
"""

from config import OUTPUT_DIR
from common_plots import ensure_output_dir
from task1_data import load_data, group_counts, data_quality_table
from task2_one_sample import one_sample_tests
from task2_plots import plot_distribution, plot_group_normality
from task3_two_sample import pairwise_tests
from task3_plots import plot_pairwise_heatmap
from task4_multi_group import multiple_group_tests
from task4_plots import plot_anova_diagnostics
from task5_multivariate import manova_test, hotelling_tests, box_m_test
from task5_plots import plot_scatter_with_ellipses, plot_perch_roach_distributions


# [Shared] Save a table as CSV into the output directory.
def save_csv(table, filename, with_index=False):
    table.to_csv(ensure_output_dir() / filename, index=with_index)


# [Shared] Save text into the output directory.
def save_text(text, filename):
    (ensure_output_dir() / filename).write_text(text, encoding="utf-8")


# [Task 1] Load data and save group sizes and data quality checks.
def run_task1():
    data = load_data()
    save_csv(group_counts(data), "pocet_ryb.csv", with_index=True)
    save_csv(data_quality_table(data), "kontrola_dat.csv")
    return data


# [Task 2] One-sample tests + distribution and normality figures.
def run_task2(data):
    save_csv(one_sample_tests(data), "jednovyberove.csv")
    plot_distribution(data)
    plot_group_normality(data)


# [Task 3] Two-sample tests + p-value heatmap. Returns the table for reuse.
def run_task3(data):
    pairwise = pairwise_tests(data)
    save_csv(pairwise, "dvouvyberove.csv")
    plot_pairwise_heatmap(pairwise)
    return pairwise


# [Task 4] ANOVA/Tukey and Kruskal–Wallis/Dunn + residual diagnostics.
def run_task4(data):
    anova, tukey, kruskal, dunn = multiple_group_tests(data)
    save_csv(anova, "anova_diagnostika.csv")
    save_csv(tukey, "tukey.csv")
    save_csv(kruskal, "kruskal_wallis.csv")
    save_csv(dunn, "dunn_holm.csv")
    plot_anova_diagnostics(data)


# [Task 5] MANOVA, Hotelling T^2, Box's M + scatter plot with ellipses.
def run_task5(data):
    save_text(str(manova_test(data)), "manova.txt")
    save_csv(hotelling_tests(data), "hotelling_parove.csv")
    save_text(str(box_m_test(data)), "box_m.txt")
    plot_scatter_with_ellipses(data)
    plot_perch_roach_distributions(data)


# [Shared] Print a short summary to the console.
def print_summary(data):
    print("Počty pozorování:")
    print(group_counts(data).to_string())
    print(f"\nVýsledky jsou uloženy v adresáři {OUTPUT_DIR}/")


# Run the tasks in assignment order.
def main():
    data = run_task1()
    run_task2(data)
    run_task3(data)
    run_task4(data)
    run_task5(data)
    print_summary(data)


if __name__ == "__main__":
    main()
