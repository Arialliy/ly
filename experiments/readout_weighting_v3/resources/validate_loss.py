"""CPU-only new-loss validation, including unmodified supplied reference tests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

E = Path(__file__).resolve().parents[1]
REF = E / "reference/DeepProPlus_C_readout_v3"
OUT = E / "resources"


def run(args, cwd=E):
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd, check=True)


def main():
    run(["-m", "pytest", "-q", REF / "test_readout_weighting.py", "--import-mode=importlib",
         "--junitxml="+str(OUT / "REFERENCE_TESTS.xml"), "-p", "no:cacheprovider"], cwd=REF)
    source = E.parents[1] / "experiments/track_contrast_v2/resources/DEIM_deim_criterion.py"
    run([REF / "check_official_kernel.py", "--source", source,
         "--output", OUT / "OFFICIAL_KERNEL_REFERENCE.json"], cwd=REF)
    spec = importlib.util.spec_from_file_location("_v3_original_reference", REF / "readout_weighting.py")
    ref = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ref; spec.loader.exec_module(ref)
    spec = importlib.util.spec_from_file_location("_v3_local_adapter", E / "train_modules/readout_weighting.py")
    local = importlib.util.module_from_spec(spec); sys.modules[spec.name] = local; spec.loader.exec_module(local)
    rows = []
    for level in [-20., -40., -80., -1000., -10000., -100000.]:
        positive = torch.zeros((1, 8), dtype=torch.float32)
        negative = torch.full((1, 8), level, dtype=torch.float32)
        a = ref.readout_terms(positive, negative, mode="matched_confidence")
        b = local.readout_terms(positive, negative, mode="matched_confidence")
        rows.append({"logit": level, "reference_weight_min": a.negative_weight.min().item(),
                     "reference_weight_max": a.negative_weight.max().item(),
                     "reference_equal_score_max_error": (a.negative_weight-1).abs().max().item(),
                     "local_equal_score_max_error": (b["negative_weight"]-1).abs().max().item()})
    (OUT / "NUMERICAL_REVIEW.json").write_text(json.dumps({
        "scope": "Equal-score N groups should reduce to unit; float32 input, no epsilon/clamp.",
        "observations": rows,
        "reference_unchanged": True,
        "adaptation": "Detached normalization arithmetic and combined log weights use float64 before cast to original loss compute dtype.",
    }, indent=2)+"\n", encoding="utf-8")
    run(["-m", "pytest", "-q", E / "tests/test_readout_weighting.py", "--import-mode=importlib",
         "--junitxml="+str(OUT / "INTEGRATION_TESTS.xml"), "-p", "no:cacheprovider"])
    xml = {}
    for name in ["REFERENCE_TESTS.xml", "INTEGRATION_TESTS.xml"]:
        suites = list(ET.parse(OUT/name).getroot().iter("testsuite"))
        xml[name] = {k: sum(int(x.attrib[k]) for x in suites) for k in ["tests", "failures", "errors", "skipped"]}
    paths = [REF/"readout_weighting.py", REF/"test_readout_weighting.py", REF/"check_official_kernel.py",
             E/"train_modules/readout_weighting.py", E/"tests/test_readout_weighting.py", OUT/"official_kernel.py", source]
    summary = {"status": "PASS", "device": "CPU", "torch_version": torch.__version__,
               "tests": xml, "reference_tests_executed_locally": True,
               "official_original_methods_executed": ["loss_labels_mal", "_get_src_permutation_idx"],
               "official_scope": "All-background MAL only, gamma=2, mal_alpha=None, matched equal-mean reduction; not full MAL or detector reproduction.",
               "official_commit": "09d35d53d39ee3145a1e61e3a989b28b9468d1dd",
               "commit_evidence": "Pinned v2 DEIM_REFERENCE.json and verified criterion SHA256; the raw supplied checker git lookup is not used as author provenance.",
               "source_sha256": {str(p.relative_to(E.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / "LOSS_VALIDATION.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
