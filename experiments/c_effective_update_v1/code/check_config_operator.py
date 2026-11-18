"""CPU-only synthetic checks of the new wrapper; no model/training job."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

E = Path(__file__).resolve().parents[1]


def main():
    xml = E / "resources/CONFIG_OPERATOR_TESTS.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(E/"tests/test_config_operator.py"),
                    "--junitxml="+str(xml), "-p", "no:cacheprovider"], check=True)
    suites = list(ET.parse(xml).getroot().iter("testsuite"))
    totals = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    assert totals["failures"] == totals["errors"] == totals["skipped"] == 0
    result = {"status": "PASS_CPU_SYNTHETIC_ONLY", "device": "CPU", "torch": torch.__version__,
        "tests": totals, "real_checkpoint_ConFIG_step_executed": False,
        "training_executed": False, "scope": "Official normal-case direction, fixed-coordinate None behavior, Adam zero-vs-None semantics, numerical fallback and malformed input checks only.",
        "source_sha256": {str(p.relative_to(E)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [E/"code/config_operator.py", E/"tests/test_config_operator.py", E/"resources/config_official/SOURCE_MANIFEST.json"]}}
    (E/"resources/CONFIG_OPERATOR_CHECK.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
