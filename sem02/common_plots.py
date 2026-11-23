"""Small plotting helpers shared by several assignment tasks."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from config import SPECIES, OUTPUT_DIR

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


# [Shared: tasks 2, 3, 4, 5] One fixed colour per species, so all figures match.
def species_colors():
    return dict(zip(SPECIES, sns.color_palette("colorblind", len(SPECIES))))


# [Shared: tasks 2, 3, 4, 5] Make sure the output directory exists.
def ensure_output_dir():
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    return output


# [Shared: tasks 2, 3, 4, 5] Tidy the layout, save the figure as PNG and close it.
def save_figure(fig, filename, dpi=240):
    fig.tight_layout()
    fig.savefig(ensure_output_dir() / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
