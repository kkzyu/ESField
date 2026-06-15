from pathlib import Path
import sys

from natsort import natsorted
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.Draw import MolsMatrixToGridImage
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

RDLogger.DisableLog("rdApp.*")


def read_sdf(sdf_fp):
    suppl = Chem.SDMolSupplier(str(sdf_fp), removeHs=False, sanitize=False)
    return [mol for mol in suppl]


def mols2grid(mols3d, width, score_dict=None):
    legends = []
    mols = []
    skipped_ids = []
    for i, mol in enumerate(mols3d):
        try:
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            _mol = Chem.MolFromSmiles(sm, sanitize=False)
            AllChem.Compute2DCoords(_mol)
        except Exception:
            sm = Chem.MolToSmiles(mol)
            _mol = mol
        if mol:
            legends.append(f"{sm}\n")
            mols.append(_mol)

    if score_dict:
        for key, vals in score_dict.items():
            for i in range(len(mols3d)):
                if i in skipped_ids:
                    continue
                legend_str = legends[i]
                val = list(vals)[i]
                if isinstance(val, float):
                    val = round(val, 3)
                legend_str += f"{key}: {val}\n"
                legends[i] = legend_str
                # legends.append(legend_str)
    mols_matrix = np.array(mols).reshape(width, -1).tolist()
    lg_matrix = np.array(legends).reshape(width, -1).tolist()
    return (
        MolsMatrixToGridImage(
            molsMatrix=mols_matrix, legendsMatrix=lg_matrix, subImgSize=(300, 250)
        ),
        mols,
    )


def calc_energy(mol: Chem.Mol, per_atom: bool = True) -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield
    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.
    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False
    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """
    mol_copy = Chem.Mol(mol)
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        energy = ff.CalcEnergy()
        energy = energy / mol.GetNumAtoms() if per_atom else energy
    except Exception:
        energy = None
    return energy


def optimise_mol(mol: Chem.Mol, max_iters: int = 200) -> Chem.Mol:
    """Optimise the conformation of an RDKit molecule

    Only the first (0th index) conformer within the molecule is optimised. The molecule is copied so the original
    is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm

    Returns:
        Chem.Mol: Optimised molecule or None if the molecule could not be optimised within the given number of
                iterations
    """

    mol_copy = Chem.Mol(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except Exception:
        return None

    return mol_copy


def get_qed(ligand_path):
    qeds = []
    protein_counter = 0
    try:
        ligands = read_sdf(ligand_path)
    except BaseException:
        ligands = []

    for ligand in ligands:
        try:
            qeds.append(qed(ligand))
        except Exception:
            continue
    if len(qeds):
        protein_counter = 1
    return {"qeds": qeds, "protein_counter": protein_counter}


def get_strain_energy(ligand_path):
    try:
        ligands = read_sdf(ligand_path)
    except BaseException:
        ligands = []

    energy, energy_opt, strain_energy = [], [], []
    for ligand in ligands:
        try:
            lopt = optimise_mol(ligand)
            e = calc_energy(ligand)
            eopt = calc_energy(lopt)
            energy.append(e)
            energy_opt.append(eopt)
            strain_energy.append(e - eopt)
        except Exception:
            continue
    return {"energy": energy, "energy_opt": energy_opt, "strain_energy": strain_energy}


def get_interaction_count(ligand_path):
    # Try different possible interaction file names
    possible_files = [
        ligand_path.parent / "interactions.csv",
        ligand_path.parent / "mols_interactions.csv",
        ligand_path.parent / "opt_mols_interactions.csv",
    ]

    for interaction_fp in possible_files:
        if interaction_fp.exists():
            try:
                inter_df = pd.read_csv(interaction_fp)
                return {"interaction_count": sum(inter_df["interactions"].values)}
            except BaseException:
                continue

    return {"interaction_count": 0}


def get_posebuster(ligand_path):
    pb_fp = ligand_path.parent / "posebuster.csv"
    try:
        pb_df = pd.read_csv(pb_fp)

        pb_cols = [
            "mol_pred_loaded",
            "mol_cond_loaded",
            "sanitization",
            "inchi_convertible",
            "all_atoms_connected",
            "bond_lengths",
            "bond_angles",
            "internal_steric_clash",
            "aromatic_ring_flatness",
            "non-aromatic_ring_non-flatness",
            "double_bond_flatness",
            "internal_energy",
            "protein-ligand_maximum_distance",
            "minimum_distance_to_protein",
            "minimum_distance_to_organic_cofactors",
            "minimum_distance_to_inorganic_cofactors",
            "minimum_distance_to_waters",
            "volume_overlap_with_protein",
            "volume_overlap_with_organic_cofactors",
            "volume_overlap_with_inorganic_cofactors",
            "volume_overlap_with_waters",
        ]

        if len(pb_df):
            pb_df = pb_df[pb_cols]
            pb_df = pb_df.map(lambda x: x if isinstance(x, bool) else False)
            return {
                "pbr": pb_df.values.all(-1).sum(),
                "pbr_ratio": pb_df.values.all(-1).tolist(),
            }
        return {"pbr": 0, "pbr_ratio": []}

    except BaseException:
        return {"pbr": 0, "pbr_ratio": []}


def get_glide_score(ligand_path, SCORE_COL_NAME="r_i_docking_score"):
    score_fp = ligand_path.parent / "glide_in.csv"

    scores, neg_count, neg_ratio = None, None, None

    if score_fp.exists():
        score_df = pd.read_csv(score_fp)
        neg_df = score_df[score_df[SCORE_COL_NAME] < 0]
        neg_count = len(neg_df)
        valid_score_df = score_df[score_df[SCORE_COL_NAME] < 10000]
        scores = (valid_score_df[SCORE_COL_NAME]).to_list()
        if len(valid_score_df) != 0:
            neg_ratio = sum(valid_score_df[SCORE_COL_NAME] < 0) / len(valid_score_df)
    return {"scores": scores, "neg_count": neg_count, "neg_ratio": neg_ratio}


def get_better_than_native(ligand_path, pdb_dir, SCORE_COL_NAME="r_i_docking_score"):
    pdb_id = ligand_path.parent.name
    score_fp = ligand_path.parent / "glide_in.csv"
    native_score_fp = pdb_dir / pdb_id / "score.csv"

    btn_count = 0

    if score_fp.exists():
        if not native_score_fp.exists():
            # Determine which model this is from based on the path
            model_type = "Unknown"
            if "edm" in str(ligand_path):
                model_type = "EDM"
            elif "semlaflow" in str(ligand_path):
                model_type = "SemlaFlow"
            print(
                f"SKIPPED: Protein {pdb_id} missing native score data for {model_type} model"
            )
            return {"btn_count": 0}

        try:
            score_df = pd.read_csv(score_fp)
            native_score_df = pd.read_csv(native_score_fp)
            native_score = native_score_df[SCORE_COL_NAME].item()
            neg_df = score_df[score_df[SCORE_COL_NAME] < 0]
            btn_count = sum(neg_df[SCORE_COL_NAME] < native_score)
        except Exception as e:
            # Determine which model this is from based on the path
            model_type = "Unknown"
            if "edm" in str(ligand_path):
                model_type = "EDM"
            elif "semlaflow" in str(ligand_path):
                model_type = "SemlaFlow"
            print(
                f"SKIPPED: Protein {pdb_id} has corrupted data for {model_type} model: {e}"
            )
            return {"btn_count": 0}
    return {"btn_count": btn_count}


def get_vina_score(ligand_path):
    scores = []
    total_scores = 0
    neg_count = 0
    for vscore_txt in (ligand_path.parent / "mols_vina").rglob("*.txt"):
        try:
            score = float(
                [
                    line.strip()
                    for line in open(vscore_txt).readlines()
                    if len(line.strip()) > 0
                ][0]
                .split("Affinity: ")[1]
                .split(" ")[0]
            )
            if score < 10000:
                scores.append(score)
                neg_count += score < 0
                total_scores += 1
        except Exception:
            pass
    if len(scores):
        return {
            "scores": scores,
            "neg_count": neg_count,
            "neg_ratio": neg_count / max(1, total_scores),
        }
    return {"scores": None, "neg_count": None, "neg_ratio": None}


def get_diversity(ligand_path, reference_path):
    sames = []
    for lp in tqdm(
        natsorted(Path(ligand_path).rglob("*/generated_ligands_no_frags.sdf"))
    ):
        ng_ligand_path = Path(reference_path) / lp.parent.name / lp.name
        if not ng_ligand_path.exists():
            continue
        l1, l2 = [], set()
        try:
            with Chem.SDMolSupplier(str(lp), removeHs=True, sanitize=True) as suppl:
                for mol in suppl:
                    if mol:
                        l1.append(Chem.MolToSmiles(mol))
        except BaseException:
            pass
        try:
            with Chem.SDMolSupplier(
                str(ng_ligand_path), removeHs=True, sanitize=True
            ) as suppl:
                for mol in suppl:
                    if mol:
                        l2.add(Chem.MolToSmiles(mol))
        except BaseException:
            pass

        sames += [smiles in l2 for smiles in l1]
    return 1 - float(np.mean(sames))


def get_head_foot_table1(model_name):
    table_1_head = (
        r"""\begin{table}[!ht]
      \caption{Docking Score Evaluation for the """
        + model_name
        + r""" model. VR $<$ 0: negative vina score ratio, GR $<$ 0: negative docking score ratio; VS: vina score, GS: glide score (mean docking score).}
      \label{tab:results:"""
        + model_name.lower()
        + r"""-docking-score-evaluation}
      \centering
      \setlength\tabcolsep{6pt}
      \begin{small}
        \begin{sc}
          \begin{tabular}{lcccc}
            \toprule
            Method & $\text{VR} < 0$ & $\text{GR} < 0$ & VS & GS \\
            \midrule
    """
    )
    table_1_foot = r"""
            \bottomrule
          \end{tabular}
        \end{sc}
      \end{small}
    \end{table}"""
    return table_1_head, table_1_foot


def get_head_foot_table2(model_name):
    table_2_head = (
        r"""\begin{table}[!ht]
      \caption{Molecule Quality Metrics for the """
        + model_name
        + r""" model. QED: quantitative estimate of drug-likeness, PBR: PoseBuster pass ratio, BNC: better-than-native count.}
      \label{tab:results:"""
        + model_name.lower()
        + r"""-docking-score-evaluation}
      \centering
      \setlength\tabcolsep{5pt}
      \begin{small}
        \begin{sc}
          \begin{tabular}{lcccccc}
            \toprule
            Method & QED & PBR & BNC & Valid & \# Interactions & Strain Energy \\
            \midrule
    """
    )
    table_2_foot = r"""
            \bottomrule
          \end{tabular}
        \end{sc}
      \end{small}
    \end{table}"""
    return table_2_head, table_2_foot


def get_row_table1(gen_mols_dir, row_name, prefix="", suffix="_no_frags", rmsd=False):
    ligand_paths = natsorted(
        Path(gen_mols_dir).rglob(f"*/{prefix}generated_ligands{suffix}.sdf")
    )
    table_1_row = r"{}       & {:.2f}\% & {:.2f}\% & {:.2f} & {:.2f} \\"
    vina_stats = {"scores": [], "neg_count": [], "neg_ratio": []}
    glide_stats = {"scores": [], "neg_count": [], "neg_ratio": []}

    for ligand_fp in tqdm(ligand_paths):
        _vina_stats = get_vina_score(ligand_fp)
        _glide_stats = get_glide_score(ligand_fp)
        for k, v in _vina_stats.items():
            if v is not None:
                vina_stats[k].append(v)
        for k, v in _glide_stats.items():
            if v is not None:
                glide_stats[k].append(v)

    r1 = 100 * sum(vina_stats["neg_count"]) / (len(vina_stats["neg_count"]) * 128)
    r2 = 100 * sum(glide_stats["neg_count"]) / (len(glide_stats["neg_count"]) * 128)
    r3 = np.concatenate(vina_stats["scores"]).mean()
    r4 = np.concatenate(glide_stats["scores"]).mean()
    return table_1_row.format(row_name, r1, r2, r3, r4)


def get_row_table2(
    gen_mols_dir, pdb_dir, row_name, prefix="", suffix="_no_frags", rmsd=False
):
    ligand_paths = natsorted(
        Path(gen_mols_dir).rglob(f"*/{prefix}generated_ligands{suffix}.sdf")
    )
    table_2_row = r"{}       & {:.2f} & {:.2f}\% & {:d} & {:.2f}\% & {:.2f} & {:.2f} \\"

    stats = {
        "qeds": [],
        "protein_counter": 0,
        "pbr": 0,
        "btn_count": 0,
        "interaction_count": 0,
        "energy": [],
        "energy_opt": [],
        "strain_energy": [],
    }
    stats["pbr_ratio"] = []
    for ligand_fp in tqdm(ligand_paths):
        _qed_stats = get_qed(ligand_fp)
        _pb_stats = get_posebuster(ligand_fp)
        _btn_stats = get_better_than_native(ligand_fp, pdb_dir)
        _int_stats = get_interaction_count(ligand_fp)
        _eg_stats = get_strain_energy(ligand_fp)
        {}
        for k, v in (
            _qed_stats | _pb_stats | _btn_stats | _int_stats | _eg_stats
        ).items():
            stats[k] += v
    r1 = np.mean(stats["qeds"])
    r2 = 100 * np.sum(stats["pbr"]) / (stats["protein_counter"] * 128)
    print(r2, 100 * np.mean(stats["pbr_ratio"]))
    r3 = stats["btn_count"]
    r4 = 100 * len(stats["qeds"]) / (stats["protein_counter"] * 128)
    r5 = stats["interaction_count"] / (stats["protein_counter"] * 128)
    r6 = np.nanmean(stats["strain_energy"])
    return table_2_row.format(row_name, r1, r2, r3, r4, r5, r6)


if __name__ == "__main__":
    result_dir = Path(sys.argv[1])
    pdb_dir = Path(sys.argv[2])

    result_suffix = {
        "EDM": {
            "No guidance": "edm_250/from_pretrain",
            "No guidance + Opt": "edm_250_post_optimization/from_pretrain",
            "Guidance": "edm_250/from_pretrain_with_guidance",
            "Guidance + Opt": "edm_250_post_optimization/from_pretrain_with_guidance",
        },
        "SemlaFlow": {
            "No guidance": "semlaflow/from_pretrain",
            "No guidance + Opt": "semlaflow_post_optimization/from_pretrain",
            "Guidance": "semlaflow/from_pretrain_with_guidance",
            "Guidance + Opt": "semlaflow_post_optimization/from_pretrain_with_guidance",
        },
    }

    for model_type, model_runs in result_suffix.items():
        table_1_head, table_1_foot = get_head_foot_table1(model_type)
        table_2_head, table_2_foot = get_head_foot_table2(model_type)

        rows_t1, rows_t2 = [], []
        for model_run, suff in model_runs.items():
            rows_t1.append(get_row_table1(result_dir / suff, model_run))
            rows_t2.append(get_row_table2(result_dir / suff, pdb_dir, model_run))

        print(table_1_head + "\n".join(rows_t1) + table_1_foot)
        print(table_2_head + "\n".join(rows_t2) + table_2_foot)

    print("\n" + "=" * 60)
    print("SUMMARY: Table generation completed successfully!")
    print("Note: Any proteins with missing native score data were skipped.")
    print("=" * 60)
