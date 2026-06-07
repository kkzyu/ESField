"""Create a simple site-map scatter plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from site_detection.site_schema import SiteMap

COLORS = {
    "high_energy_water": "tab:red",
    "stable_water": "tab:blue",
    "hydrophobic_cavity": "gold",
}


def plot_site_map(site_map: SiteMap, output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_plot_fallback_csv(site_map, target.with_suffix(".csv"))
        print(f"matplotlib not installed; wrote fallback CSV: {target.with_suffix('.csv')}")
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for site_type in sorted({site.site_type for site in site_map.sites}):
        sites = [site for site in site_map.sites if site.site_type == site_type]
        ax.scatter(
            [site.center[0] for site in sites],
            [site.center[1] for site in sites],
            [site.center[2] for site in sites],
            s=[max(30.0, site.radius * 80.0) for site in sites],
            c=COLORS.get(site_type, "gray"),
            label=site_type,
            alpha=0.75,
            edgecolors="black",
        )
    ax.set_title(f"{site_map.protein_id} / {site_map.ligand_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _write_plot_fallback_csv(site_map: SiteMap, output: Path) -> None:
    with output.open("wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_id", "site_type", "x", "y", "z", "radius", "score", "confidence"])
        writer.writeheader()
        for site in site_map.sites:
            writer.writerow(
                {
                    "site_id": site.site_id,
                    "site_type": site.site_type,
                    "x": site.center[0],
                    "y": site.center[1],
                    "z": site.center[2],
                    "radius": site.radius,
                    "score": site.score,
                    "confidence": site.confidence,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an ESField site map.")
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_site_map(SiteMap.read_json(args.site_map), args.output)
    print(f"wrote site-map plot: {args.output}")


if __name__ == "__main__":
    main()

