"""Generator interface contracts and RepMolFlow command adapter."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GeneratedMolecule:
    coordinates: object
    atom_type_indices: object | None = None
    atom_type_probs: object | None = None
    metadata: dict = field(default_factory=dict)


class FlowMatchingGenerator(Protocol):
    def velocity(self, state, protein_context, t: float):
        ...

    def step(self, state, velocity, t: float, dt: float):
        ...


@dataclass(frozen=True)
class RepMolFlowCommand:
    repo_dir: Path
    model_checkpoint: Path
    output_file: Path
    n_mols: int = 1000
    max_batch_size: int = 128
    n_timesteps: int = 100
    property_name: str = "alpha"
    properties_handle_method: str = "sum"
    properties_for_sampling: float | None = None
    multiple_values_file: Path | None = None
    number_of_atoms_file: Path | None = None
    normalization_file_path: Path | None = None
    n_atoms_per_mol: int | None = None
    gpu: int = 0
    analyze: bool = False

    @property
    def resolved_repo_dir(self) -> Path:
        return resolve_repmolflow_repo_dir(self.repo_dir)

    def argv(self) -> list[str]:
        repo_dir = self.resolved_repo_dir
        args = [
            "python",
            str(repo_dir / "sample_condition.py"),
            "--model_checkpoint",
            str(self.model_checkpoint),
            "--n_mols",
            str(self.n_mols),
            "--max_batch_size",
            str(self.max_batch_size),
            "--n_timesteps",
            str(self.n_timesteps),
            "--property_name",
            self.property_name,
            "--properties_handle_method",
            self.properties_handle_method,
            "--output_file",
            str(self.output_file),
            "--gpu",
            str(self.gpu),
        ]
        if self.properties_for_sampling is not None:
            args.extend(["--properties_for_sampling", str(self.properties_for_sampling)])
        if self.multiple_values_file is not None:
            args.extend(["--multiple_values_file", str(self.multiple_values_file)])
        if self.number_of_atoms_file is not None:
            args.extend(["--number_of_atoms", str(self.number_of_atoms_file)])
        if self.normalization_file_path is not None:
            args.extend(["--normalization_file_path", str(self.normalization_file_path)])
        if self.n_atoms_per_mol is not None:
            args.extend(["--n_atoms_per_mol", str(self.n_atoms_per_mol)])
        if self.analyze:
            args.append("--analyze")
        return args

    def shell_command(self) -> str:
        return " ".join(shlex.quote(arg) for arg in self.argv())

    def run(self, *, dry_run: bool = True) -> subprocess.CompletedProcess[str] | None:
        if dry_run:
            print(self.shell_command())
            return None
        return subprocess.run(self.argv(), cwd=self.resolved_repo_dir, text=True, check=True)


def resolve_repmolflow_repo_dir(repo_dir: str | Path) -> Path:
    """Resolve zip extraction layouts to the directory that contains sample_condition.py."""

    root = Path(repo_dir)
    if (root / "sample_condition.py").exists():
        return root
    nested = root / "RepMolFlow-main"
    if (nested / "sample_condition.py").exists():
        return nested
    raise FileNotFoundError(
        "cannot find RepMolFlow sample_condition.py; pass the directory containing it "
        f"or its zip-extracted parent. Checked: {root} and {nested}"
    )


def explain_repmolflow_scope() -> str:
    return (
        "RepMolFlow/PropMolFlow 当前代码是 QM9 property-conditioned flow matching 示例，"
        "不是 pocket-conditioned SBDD 生成器。ESField 代码提供命令适配器和通用 velocity guidance hook；"
        "真正把 site gradient 插入采样循环时，需要在目标生成器每个 ODE step 取得 coordinates、base_velocity "
        "以及 atom_type_indices 或 atom_type_probs。"
    )
