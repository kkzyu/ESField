"""Guided generation entrypoint scaffold for flow-matching models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.adapt_generator_interface import explain_repmolflow_scope


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ESField guided generation run metadata.")
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--potential-checkpoint", required=True)
    parser.add_argument("--base-generator", default="custom_flow_matching")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-max", type=float, default=0.1)
    parser.add_argument("--guidance-start", type=float, default=0.35)
    parser.add_argument("--guidance-end", type=float, default=0.90)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_generator": args.base_generator,
        "site_map": args.site_map,
        "potential_checkpoint": args.potential_checkpoint,
        "guidance": {
            "lambda_max": args.lambda_max,
            "guidance_start": args.guidance_start,
            "guidance_end": args.guidance_end,
            "grad_clip": args.grad_clip,
        },
        "adapter_note": explain_repmolflow_scope(),
    }
    (output_dir / "guided_generation_plan.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote guided generation plan:", output_dir / "guided_generation_plan.json")
    if not args.dry_run:
        raise NotImplementedError(
            "Guided generation execution is not yet implemented. "
            "To prepare metadata only, pass --dry-run. "
            "To run guided generation, patch the target flow-matching sampler with "
            "guidance.flow_matching_guidance.apply_site_guidance_to_velocity "
            "inside its ODE stepping loop."
        )


if __name__ == "__main__":
    main()
