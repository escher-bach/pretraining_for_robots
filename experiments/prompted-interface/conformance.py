"""Run the prompted-interface fixture and emit one canonical JSON artifact."""

from __future__ import annotations

import argparse
import json
import pathlib
import unittest

import test_protocol
from protocol import make_bundle, public_projection, target_supervision_projection, validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    result = unittest.defaultTestLoader.loadTestsFromModule(test_protocol)
    report = unittest.TextTestRunner(verbosity=2).run(result)
    if not report.wasSuccessful():
        return 1

    bundle = make_bundle()
    validate(bundle)
    artifact = {
        "public_episode": public_projection(bundle),
        "supervision": target_supervision_projection(bundle),
        "private_receipt": bundle["private_receipt"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
