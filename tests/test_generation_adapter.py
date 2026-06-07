from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation.adapt_generator_interface import RepMolFlowCommand, resolve_repmolflow_repo_dir  # noqa: E402


class GenerationAdapterTest(unittest.TestCase):
    def test_resolves_nested_repmolflow_zip_layout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            outer = Path(tmp_dir) / "RepMolFlow-main"
            inner = outer / "RepMolFlow-main"
            inner.mkdir(parents=True)
            (inner / "sample_condition.py").write_text("print('toy')\n", encoding="utf-8")

            resolved = resolve_repmolflow_repo_dir(outer)
            command = RepMolFlowCommand(
                repo_dir=outer,
                model_checkpoint=Path("checkpoint.ckpt"),
                output_file=Path("out.sdf"),
            )
            command_text = command.shell_command()

            self.assertEqual(resolved, inner)
            self.assertIn(str(inner / "sample_condition.py"), command_text)


if __name__ == "__main__":
    unittest.main()
