#!/usr/bin/env python3
"""
Generate Markdown tables from unified experiment results.

Reads SDF directories from experiments/master_experiments/unified/
and generates the comparison matrix and ablation tables.
"""

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import evaluate_condition

UNIFIED_DIR = ROOT / "experiments/master_experiments/unified"
SITE_MAP_DIR = ROOT / "experiments/targetdiff_replication/site_maps"
DATA_ROOT = Path("/root/autodl-tmp/data")

POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "ref_atoms": 26},
}


def get_protein_path(pocket_id: str) -> str:
    cfg = POCKET_CONFIG[pocket_id]
    return str(DATA_ROOT / "PDB/P-L" / cfg["year"] / pocket_id / f"{pocket_id}_protein.pdb")


def get_site_map_path(pocket_id: str) -> str:
    p = SITE_MAP_DIR / f"{pocket_id}_site_map.json"
    return str(p) if p.exists() else None


def eval_one(exp_dir: Path, pocket: str) -> dict:
    """Run evaluation on one experimental condition."""
    sdf_dir = exp_dir / "sdfs"
    if not sdf_dir.exists() or not list(sdf_dir.glob("*.sdf")):
        return {"error": "No SDF files", "sdf_dir": str(sdf_dir)}

    protein_pdb = get_protein_path(pocket)
    site_map = get_site_map_path(pocket)

    result = evaluate_condition(
        sdf_dir=str(sdf_dir),
        protein_pdb=protein_pdb,
        site_json=site_map,
        run_vina=False,
    )
    return result


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from evaluation result."""
    m = {}

    # Validity
    v = result.get("validity", {})
    m["valid_total"] = v.get("n_total", 0)
    m["valid_n"] = v.get("n_valid", 0)
    m["valid_rate"] = v.get("validity_rate", 0)

    # Strain
    s = result.get("strain_energy", {})
    m["strain_mean"] = s.get("strain_energy_mean")
    m["strain_std"] = s.get("strain_energy_std")

    # Clash
    c = result.get("clash_score", {})
    m["clash_mean"] = c.get("clash_score_mean")

    # PBR
    p = result.get("pbr", {})
    m["pbr_mean"] = p.get("pbr_mean")

    # QED
    q = result.get("qed", {})
    m["qed_mean"] = q.get("qed_mean")

    # SA Score
    sa = result.get("sa_score", {})
    m["sa_mean"] = sa.get("sa_score_mean")

    # DirectOcc HEW
    do = result.get("direct_occ", {})
    m["docc_hew"] = do.get("direct_occ_hew")

    # DirectOcc SW
    m["docc_sw"] = do.get("direct_occ_sw")

    # Diversity
    div = result.get("diversity", {})
    m["diversity"] = div.get("diversity_mean")

    return m


def fmt(val, fmt_spec=".1f"):
    """Format a value for table display."""
    if val is None:
        return "---"
    if isinstance(val, float):
        if fmt_spec == ".1f":
            return f"{val:.1f}"
        elif fmt_spec == ".3f":
            return f"{val:.3f}"
        elif fmt_spec == ".1%" or fmt_spec == ".0%":
            return f"{val*100:.0f}%"
        return f"{val}"
    return str(val)


def generate_main_table(results: dict, pocket: str) -> str:
    """Generate the main 5-condition comparison table."""
    guidance_order = ["unguided", "hard_fix", "lai_soft_fix", "badger_proxy", "kag"]
    labels = {
        "unguided": "Unguided",
        "hard_fix": "Hard-Fix (Neg. Ctrl)",
        "lai_soft_fix": "Lai + Soft-Fix",
        "badger_proxy": "BADGER (Proxy)",
        "kag": "KAG (Ours)",
    }

    lines = []
    lines.append(f"### Table 2 (Revised): Main Comparison Matrix — {pocket} (N=50)")
    lines.append("")
    lines.append("| Method | Valid | Strain/atom | Clash | DOcc_HEW | QED | SA | PBR |")
    lines.append("|--------|-------|-------------|-------|----------|-----|-----|-----|")

    for g in guidance_order:
        if g not in results:
            continue
        m = results[g]
        label = labels.get(g, g)
        valid_str = f"{m.get('valid_n', '?')}/{m.get('valid_total', '?')}"
        strain_str = fmt(m.get("strain_mean"), ".1f")
        if m.get("strain_mean") and m["strain_mean"] > 10000:
            strain_str = f"{m['strain_mean']:.1e}"
        clash_str = fmt(m.get("clash_mean"), ".3f")
        docc_str = fmt(m.get("docc_hew"), ".0%")
        qed_str = fmt(m.get("qed_mean"), ".3f")
        sa_str = fmt(m.get("sa_mean"), ".1f")
        pbr_str = fmt(m.get("pbr_mean"), ".3f")
        lines.append(f"| {label} | {valid_str} | {strain_str} | {clash_str} | {docc_str} | {qed_str} | {sa_str} | {pbr_str} |")

    lines.append("")
    return "\n".join(lines)


def main():
    pocket = "3mfw"

    # Collect experiment directories
    results = {}
    for exp_dir in sorted(UNIFIED_DIR.glob(f"{pocket}_*")):
        name = exp_dir.name.replace(f"{pocket}_", "", 1)
        print(f"Evaluating: {name}...")
        r = eval_one(exp_dir, pocket)
        m = extract_metrics(r)
        m["guidance_name"] = name
        results[name] = m
        print(f"  Valid: {m['valid_n']}/{m['valid_total']}, "
              f"Strain: {fmt(m['strain_mean'])}, "
              f"DOcc_HEW: {fmt(m['docc_hew'], '.0%')}")

    # Generate main table
    print("\n" + "=" * 70)
    print(generate_main_table(results, pocket))

    # Save raw results
    out_path = UNIFIED_DIR / "evaluation_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
