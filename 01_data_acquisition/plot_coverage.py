"""Stage 1c: year x month heatmap of mean clear Landsat observations per pixel.

Reads data/raw/coverage_log.csv, writes docs/figures/coverage_heatmap.png.
This figure decides how monsoon months are handled in Stage 2.
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C


def main() -> None:
    df = pd.read_csv(C.COVERAGE_CSV)
    grid = df.pivot(index="year", columns="month", values="mean_clear_obs")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(grid.index)), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=7)
    ax.set_title("Mean clear Landsat 8/9 observations per pixel, by month")
    fig.colorbar(im, ax=ax, label="clear observations per pixel")

    out = C.ROOT / "docs" / "figures" / "coverage_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
