"""Merge, filter, and rank ESField site maps."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from site_detection.site_schema import Site, SiteMap
from utils.geometry import distance


TYPE_PRIORITY = {"high_energy_water": 3, "stable_water": 2, "hydrophobic_cavity": 1}


def merge_site_maps(
    site_maps: Iterable[SiteMap],
    *,
    merge_distance: float = 1.0,
    max_sites: int = 20,
    min_radius: float = 0.5,
    max_radius: float = 3.5,
    min_confidence: float = 0.0,
) -> SiteMap:
    maps = list(site_maps)
    if not maps:
        raise ValueError("merge_site_maps requires at least one SiteMap")
    first = maps[0]
    filtered = [
        site
        for site_map in maps
        for site in site_map.sites
        if min_radius <= site.radius <= max_radius and site.confidence >= min_confidence
    ]
    filtered.sort(key=_site_rank_key, reverse=True)

    clusters: list[list[Site]] = []
    for site in filtered:
        for cluster in clusters:
            if any(distance(site.center, other.center) <= merge_distance for other in cluster):
                cluster.append(site)
                break
        else:
            clusters.append([site])

    merged = [_merge_cluster(cluster) for cluster in clusters]
    merged.sort(key=_site_rank_key, reverse=True)
    selected = tuple(_renumber_sites(merged[:max_sites]))
    return SiteMap(
        protein_id=first.protein_id,
        ligand_id=first.ligand_id,
        pocket_center=first.pocket_center,
        coordinate_frame=first.coordinate_frame,
        sites=selected,
    )


def _merge_cluster(cluster: list[Site]) -> Site:
    if len(cluster) == 1:
        return cluster[0]
    weights = [max(site.confidence, 0.01) for site in cluster]
    total = sum(weights)
    center = (
        sum(site.center[0] * w for site, w in zip(cluster, weights)) / total,
        sum(site.center[1] * w for site, w in zip(cluster, weights)) / total,
        sum(site.center[2] * w for site, w in zip(cluster, weights)) / total,
    )
    by_type: dict[str, list[Site]] = defaultdict(list)
    for site in cluster:
        by_type[site.site_type].append(site)
    site_type = max(by_type, key=lambda key: (TYPE_PRIORITY.get(key, 0), sum(s.confidence for s in by_type[key])))
    radius = max(site.radius for site in cluster)
    score = sum(site.score * max(site.confidence, 0.01) for site in cluster) / sum(
        max(site.confidence, 0.01) for site in cluster
    )
    confidence = min(1.0, max(site.confidence for site in cluster) + 0.05 * (len(cluster) - 1))
    return Site(
        site_id=cluster[0].site_id,
        site_type=site_type,  # type: ignore[arg-type]
        center=center,
        radius=radius,
        score=score,
        confidence=confidence,
        source="+".join(sorted({site.source for site in cluster})),
        features={"merged_site_count": len(cluster), "source_site_ids": [site.site_id for site in cluster]},
    )


def _site_rank_key(site: Site) -> tuple[float, float, int]:
    return (site.confidence, abs(site.score), TYPE_PRIORITY.get(site.site_type, 0))


def _renumber_sites(sites: list[Site]) -> list[Site]:
    return [
        Site(
            site_id=index,
            site_type=site.site_type,
            center=site.center,
            radius=site.radius,
            score=site.score,
            confidence=site.confidence,
            source=site.source,
            features=site.features,
        )
        for index, site in enumerate(sites)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ESField site maps.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge-distance", type=float, default=1.0)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")

    merged = merge_site_maps(
        [SiteMap.read_json(path) for path in args.inputs],
        merge_distance=args.merge_distance,
        max_sites=args.max_sites,
        min_confidence=args.min_confidence,
    )
    print(f"merged {sum(len(SiteMap.read_json(path).sites) for path in args.inputs)} sites into {len(merged.sites)} sites")
    if not args.dry_run:
        merged.write_json(output)


if __name__ == "__main__":
    main()
