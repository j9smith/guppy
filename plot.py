import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# RTX 5060 Ti (Blackwell, GB206): GDDR7 @ 28 Gbps on a 128-bit bus.
PEAK_BANDWIDTH_GB_S = 448.0

PASS_COLOR = "#4C9A2A"
FAIL_COLOR = "#C0392B"
LINE_COLOR = "#888888"


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        results = json.load(f)
    if not results:
        raise ValueError(f"{path} contains no results")
    return results


def _plot_series(ax, results: list[dict], y_key: str, label: str | None = None):
    """Plots one metric (y_key) vs total FLOPs on `ax`. Connecting line in
    neutral gray; markers colored/shaped by correctness."""
    results = sorted(results, key=lambda r: r["M"] * r["N"] * r["K"])
    x = [2 * r["M"] * r["N"] * r["K"] for r in results]
    y = [r[y_key] for r in results]
    passed = [r["correctness_passed"] for r in results]

    ax.plot(x, y, "-", color=LINE_COLOR, linewidth=1, zorder=1, label=label)

    xs_pass = [xi for xi, p in zip(x, passed) if p]
    ys_pass = [yi for yi, p in zip(y, passed) if p]
    xs_fail = [xi for xi, p in zip(x, passed) if not p]
    ys_fail = [yi for yi, p in zip(y, passed) if not p]

    ax.scatter(xs_pass, ys_pass, color=PASS_COLOR, marker="o", s=60, zorder=2)
    ax.scatter(xs_fail, ys_fail, color=FAIL_COLOR, marker="x", s=70, zorder=2)

    for r, xi, yi in zip(results, x, y):
        ax.annotate(
            f"{r['M']}x{r['N']}x{r['K']}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            ha="center",
            color="#444444",
        )


def plot_results(results: list[dict], out_path: Path) -> None:
    n_failed = sum(1 for r in results if not r["correctness_passed"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _plot_series(ax1, results, "gflops")
    ax1.set_xscale("log")
    ax1.set_xlabel("required total FLOPs (2*M*N*K)")
    ax1.set_ylabel("GFLOPS")
    ax1.set_title("Achieved throughput vs problem size")

    _plot_series(ax2, results, "bandwidth_gb_s")
    ax2.axhline(
        PEAK_BANDWIDTH_GB_S,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"peak ({PEAK_BANDWIDTH_GB_S:.0f} GB/s)",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("required total FLOPs (2*M*N*K)")
    ax2.set_ylabel("GB/s")
    ax2.set_title("Achieved memory bandwidth vs problem size")
    ax2.legend(loc="lower right", fontsize=8)

    title = "W4A16 kernel benchmark (green = correct, red X = incorrect)"
    if n_failed:
        title += f"  —  {n_failed}/{len(results)} fixture(s) FAILED correctness"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    if n_failed:
        print(
            f"warning: {n_failed}/{len(results)} fixture(s) failed correctness "
            f"— throughput numbers for those are not meaningful yet"
        )


if __name__ == "__main__":
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results.png")

    results = load_results(results_path)
    plot_results(results, out_path)
