"""ESField guidance injection adapter for PAFlow (NeurIPS 2025).

PAFlow is a pocket-conditioned 3D flow-matching model.
This module documents the exact injection point and provides
helper code to integrate ESField site-aware energy guidance
into PAFlow's ODE sampling loop.

The injection target is:
  PAFlow: models/molopt_score_model_guide.py
  Method: ScorePosNet3D_guided_flow.sample_guided_flow_VP()
  Lines: between 703 (grad computation) and 712 (dx assembly)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation.paflow_atom_mapping import (
    PAFLOW_NUM_CLASSES,
    esfield_atom_type_to_paflow_index,
    paflow_index_to_atomic_number,
)


@dataclass(frozen=True)
class PAFlowInjectionPlan:
    """Documents the exact code changes needed in PAFlow's sampling loop."""

    target_file: str = "models/molopt_score_model_guide.py"
    target_method: str = "sample_guided_flow_VP"
    injection_after_line: int = 703

    def patch_description(self) -> str:
        return (
            "在 PAFlow 的 sample_guided_flow_VP() 方法中，第 701-703 行计算了 "
            "affinity-based pos_guidance 和 v_guidance。在 dx 组装（第 712 行）之前，"
            "插入 ESField site energy gradient：\n\n"
            "  # --- ESField site guidance (insert after line 703) ---\n"
            "  esfield_energy = esfield_guidance.total_energy(\n"
            "      ligand_xt, site_map=site_map,\n"
            "      atom_type_probs=F.softmax(preds['pred_ligand_v'], dim=-1))\n"
            "  esfield_grad = torch.autograd.grad(\n"
            "      esfield_energy, ligand_xt,\n"
            "      grad_outputs=torch.ones_like(esfield_energy),\n"
            "      retain_graph=True)[0]\n"
            "  esfield_grad = clip_by_norm(esfield_grad, max_norm=1.0)\n"
            "  esfield_lambda = esfield_lambda_schedule(t_norm)\n"
            "  pos_guidance = pos_guidance + esfield_grad * esfield_lambda\n"
            "  # --- end ESField injection ---\n"
        )


def explain_integration() -> str:
    return (
        "ESField-PAFlow 集成方案:\n\n"
        "1. PAFlow 的 sample_guided_flow_VP() 在每步 ODE 积分中:\n"
        "   - detach coordinates → requires_grad_(True)\n"
        "   - 前向传播得到 pred_ligand_pos, pred_ligand_v, pred_affinity\n"
        "   - 对 affinity log-posterior 求梯度得到 pos_guidance\n"
        "   - dx = VP_field + para_x * pos_guidance * pos_grad_w\n\n"
        "2. ESField 注入: 在 pos_guidance 计算后、dx 组装前:\n"
        "   - 用 pred_ligand_v softmax 作为 atom_type_probs\n"
        "   - 计算 total_energy(coordinates, site_map, atom_type_probs)\n"
        "   - 对坐标求梯度 → site_grad\n"
        "   - clip site_grad 并乘以 lambda_schedule(t)\n"
        "   - 加到 pos_guidance 上\n\n"
        "3. 原子类型映射:\n"
        "   ESField 11 种类型 ↔ PAFlow 13 种类型 (add_aromatic mode)\n"
        "   详见 generation/paflow_atom_mapping.py\n\n"
        "4. 采样期 site_map 来源:\n"
        "   - 从 PDBbind protein.pdb 提取 crystal water\n"
        "   - 从 fpocket 提取 hydrophobic cavity\n"
        "   - 合并后坐标对齐到 PAFlow 的口袋坐标系\n"
    )


def build_site_map_for_paflow_pocket(
    protein_pdb: str | Path,
    ligand_sdf: str | Path | None = None,
    *,
    protein_id: str,
    ligand_id: str = "reference",
    pocket_center: tuple[float, float, float] | None = None,
    fpocket_dir: str | Path | None = None,
    max_sites: int = 20,
) -> dict[str, Any]:
    """Build an ESField site map for a PAFlow pocket, ready for guidance injection.

    Returns a dict with the SiteMap JSON and integration metadata.
    """
    from site_detection.build_crystal_water_sites import (
        CrystalWaterConfig,
        build_crystal_water_site_map,
    )
    from site_detection.merge_sites import merge_site_maps
    from site_detection.parse_fpocket import FpocketParseConfig, parse_fpocket_site_map
    from site_detection.site_schema import Site, SiteMap

    water_map = build_crystal_water_site_map(
        protein_pdb,
        ligand_path=ligand_sdf,
        protein_id=protein_id,
        ligand_id=ligand_id,
        pocket_center=pocket_center,
        config=CrystalWaterConfig(max_sites=max_sites),
    )

    maps = [water_map]
    if fpocket_dir and Path(fpocket_dir).exists():
        fpocket_map = parse_fpocket_site_map(
            fpocket_dir,
            protein_id=protein_id,
            ligand_id=ligand_id,
            pocket_center=water_map.pocket_center,
            config=FpocketParseConfig(max_sites=max_sites),
        )
        maps.append(fpocket_map)

    merged = merge_site_maps(maps, merge_distance=1.0, max_sites=max_sites)

    return {
        "site_map": merged.to_dict(),
        "protein_id": protein_id,
        "ligand_id": ligand_id,
        "pocket_center": list(merged.pocket_center),
        "n_sites": len(merged.sites),
        "site_types": list(set(s.site_type for s in merged.sites)),
    }


def print_injection_guide() -> None:
    plan = PAFlowInjectionPlan()
    print(explain_integration())
    print("\n" + "=" * 70)
    print("代码注入位置:")
    print(plan.patch_description())
