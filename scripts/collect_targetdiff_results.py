#!/usr/bin/env python3
"""Collect TargetDiff results and generate LaTeX-ready tables."""
import json, sys, glob
from pathlib import Path
import numpy as np

BASE = Path("/root/ESField/experiments/master_experiments/task3_targetdiff")
SRC = Path("/root/ESField")
sys.path.insert(0, str(SRC / "src"))
sys.path.insert(0, str(SRC / "scripts"))

POCKETS = ["3mfw", "2gni", "6o4x", "2jke", "2gqn", "6phx"]
CONDS = ["unguided", "hard_fix", "kinematic"]

def collect_pocket(pocket, cond):
    """Collect all metrics for one pocket×condition."""
    d = {"pocket": pocket, "condition": cond}
    sd = BASE / pocket / cond / pocket  # nested pocket subdir from pipeline
    if not sd.exists():
        # Try alternative layout
        sd = BASE / pocket / cond
    summary_json = sd / "summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            sj = json.load(f)
        cc = sj.get("conditions", {}).get(cond, {})
        d["direct_occ_hew"] = cc.get("direct_occ", None)
        d["kpe_ratio"] = cc.get("kpe_ratio", None)
        d["n_valid"] = cc.get("n_valid", 0)
        d["n_total"] = cc.get("n_total", 0)
    else:
        d["n_valid"] = 0; d["n_total"] = 0

    # Try evaluator metrics
    sdf_dir = sd / cond / "sdf" if (sd / cond / "sdf").exists() else None
    if not sdf_dir:
        sdf_dir = sd / "sdf" if (sd / "sdf").exists() else None
    if sdf_dir and list(sdf_dir.glob("*.sdf")):
        from evaluator import (
            compute_validity_batch, compute_site_occupancy_batch,
            compute_qed_batch, compute_sa_score_batch,
        )
        combined = sd / "_c.sdf"
        with open(combined, "w") as o:
            for f in sorted(sdf_dir.glob("*.sdf")):
                with open(f) as i:
                    dd = i.read()
                    if dd.strip(): o.write(dd)
        v = compute_validity_batch(combined)
        d["validity"] = f"{v['n_valid']}/{v['n_total']}"
        qe = compute_qed_batch(combined)
        d["qed_mean"] = qe.get("qed_mean")
        sa = compute_sa_score_batch(combined)
        d["sa_mean"] = sa.get("sa_score_mean")

        # Site occupancy (HEW + SW)
        sm = SRC / "experiments/targetdiff_replication/site_maps" / f"{pocket}_site_map.json"
        if sm.exists():
            occ = compute_site_occupancy_batch(combined, str(sm))
            d["direct_occ_hew"] = occ.get("direct_occ_hew")
            d["direct_occ_sw"] = occ.get("direct_occ_sw")

    return d

# Collect
rows = []
for p in POCKETS:
    for c in CONDS:
        rows.append(collect_pocket(p, c))

# Print summary
print(f"\n{'='*100}")
print("TARGETDIFF 6-POCKET RESULTS")
print(f"{'='*100}")
hdr = f"{'Pocket':<8} {'Condition':<12} {'DOcc_HEW':>10} {'DOcc_SW':>10} {'Valid':>10} {'QED':>8} {'SA':>8} {'ρ_KPE':>10}"
print(hdr)
print("-"*100)
for r in rows:
    hew = f"{r['direct_occ_hew']:.1%}" if r['direct_occ_hew'] is not None else "--"
    sw = f"{r['direct_occ_sw']:.1%}" if r.get('direct_occ_sw') is not None else "--"
    v = r.get('validity', f"{r.get('n_valid',0)}/{r.get('n_total',0)}")
    qe = f"{r['qed_mean']:.3f}" if r.get('qed_mean') else "--"
    sa = f"{r['sa_mean']:.2f}" if r.get('sa_mean') else "--"
    kpe = f"{r['kpe_ratio']:.4f}" if r.get('kpe_ratio') is not None else "--"
    print(f"{r['pocket']:<8} {r['condition']:<12} {hew:>10} {sw:>10} {str(v):>10} {qe:>8} {sa:>8} {kpe:>10}")

# Save
out = BASE / "collected_metrics.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\nSaved: {out}")
