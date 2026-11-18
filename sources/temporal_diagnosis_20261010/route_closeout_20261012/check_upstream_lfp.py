"""Execute unmodified author LFP classes and record roundtrip failures honestly.

Exit 0: all specified checks pass. Exit 1: executed checks find functional
failures. Exit 2: a checker/dependency/device execution error occurred. Neither
failure mode patches source or silently changes a tested tensor layout.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from load_official_lfp import DEFAULT_SOURCE, load_official_lfp


def package_versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "numpy", "pytorch-wavelets", "PyWavelets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def roundtrip_case(torch, official, device, mode, channels, height, width, seed, tolerance):
    # Generate on CPU for the same input bytes on CPU and CUDA.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(2, channels, height, width, generator=generator).to(device)
    dwt = official.ConvDWT(wave="haar", mode=mode).to(device)
    idwt = official.ConvIDWT(wave="haar", mode=mode).to(device)
    with torch.no_grad():
        encoded = dwt(x)
        restored = idwt(encoded[:, :channels], encoded[:, channels:])
    shape_equal = tuple(restored.shape) == tuple(x.shape)
    finite = bool(torch.isfinite(encoded).all() and torch.isfinite(restored).all())
    error = float((restored - x).abs().max()) if shape_equal else None
    overlap_h, overlap_w = min(height, restored.shape[-2]), min(width, restored.shape[-1])
    overlap_error = float((restored[..., :overlap_h, :overlap_w]
                           - x[..., :overlap_h, :overlap_w]).abs().max())
    reasons = []
    if not shape_equal:
        reasons.append("input_shape_not_reconstructed")
    if not finite:
        reasons.append("nonfinite_output")
    if shape_equal and error > tolerance:
        reasons.append("roundtrip_error_exceeds_tolerance")
    return {
        "kind": "author_ConvDWT_ConvIDWT_roundtrip", "device": str(device),
        "mode": mode, "channels": channels, "seed": seed, "dtype": str(x.dtype),
        "input_shape": list(x.shape), "encoded_shape": list(encoded.shape),
        "restored_shape": list(restored.shape), "shape_equal": shape_equal,
        "finite": finite, "max_abs_error": error,
        "overlap_max_abs_error_diagnostic_only": overlap_error,
        "absolute_tolerance": tolerance, "passed": not reasons,
        "failure_reasons": reasons,
    }


def enhance_case(torch, official, device, channels, seed):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(1, channels, 16, 20, generator=generator).to(device).requires_grad_(True)
    model = official.wav_Enhance(in_channels=channels).to(device)
    output = model(x)
    loss = output.square().mean()
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        gradients[name] = {
            "present": parameter.grad is not None,
            "finite": bool(torch.isfinite(parameter.grad).all()) if parameter.grad is not None else None,
            "abs_max": float(parameter.grad.abs().max()) if parameter.grad is not None else None,
        }
    x_grad_finite = x.grad is not None and bool(torch.isfinite(x.grad).all())
    reasons = []
    if output.shape != x.shape:
        reasons.append("enhancement_shape_changed")
    if not bool(torch.isfinite(output).all() and torch.isfinite(loss)):
        reasons.append("nonfinite_forward")
    if not x_grad_finite:
        reasons.append("missing_or_nonfinite_input_gradient")
    if any(not item["present"] or not item["finite"] for item in gradients.values()):
        reasons.append("missing_or_nonfinite_parameter_gradient")
    return {
        "kind": "author_wav_Enhance_forward_backward", "device": str(device),
        "channels": channels, "seed": seed, "input_shape": list(x.shape),
        "output_shape": list(output.shape), "loss": float(loss.detach()),
        "input_gradient_finite": x_grad_finite,
        "input_gradient_abs_max": float(x.grad.abs().max()) if x.grad is not None else None,
        "parameter_gradients": gradients,
        "identity_required": False, "passed": not reasons, "failure_reasons": reasons,
    }


def run(args):
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(Path(args.source).resolve()), "seed": args.seed,
        "absolute_tolerance": args.tolerance, "dependency_versions": package_versions(),
        "author_classes_modified": False, "roundtrip_is_identity_check": True,
        "enhancement_is_identity_check": False, "checks": [], "checker_errors": [],
        "scope": "Synthetic operator behavior only; no detector effectiveness claim.",
    }
    try:
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        official = load_official_lfp(args.source, args.expected_sha256)
        report.update(source_sha256=official.SOURCE_SHA256,
                      selected_ast_sha256=official.SELECTED_AST_SHA256,
                      selected_class_names=list(official.SELECTED_CLASS_NAMES))
        devices = [torch.device(item) for item in args.devices]
        for device in devices:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
            for mode in ("zero", "symmetric"):
                for channels in (1, 3, 32):
                    for height, width in ((16, 20), (17, 19)):
                        case = {"device": str(device), "mode": mode, "channels": channels,
                                "height": height, "width": width}
                        try:
                            report["checks"].append(roundtrip_case(
                                torch, official, device, mode, channels, height, width,
                                args.seed, args.tolerance))
                        except Exception as exc:
                            report["checker_errors"].append({
                                "stage": "roundtrip_execution", "case": case,
                                "type": type(exc).__name__, "message": str(exc),
                                "traceback": traceback.format_exc()})
            for channels in (3, 32):
                try:
                    report["checks"].append(enhance_case(torch, official, device, channels, args.seed))
                except Exception as exc:
                    report["checker_errors"].append({
                        "stage": "enhancement_execution", "device": str(device), "channels": channels,
                        "type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc()})
    except Exception as exc:
        report["checker_errors"].append({
            "stage": "setup", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()})
    report["completed_checks"] = len(report["checks"])
    report["passed_checks"] = sum(check["passed"] for check in report["checks"])
    report["failed_checks"] = report["completed_checks"] - report["passed_checks"]
    if report["checker_errors"]:
        report["status"], exit_code = "CHECKER_ERROR", 2
    elif report["failed_checks"]:
        report["status"], exit_code = "CHECK_FAILED", 1
    else:
        report["status"], exit_code = "PASS", 0
    report["exit_code"] = exit_code
    return report, exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "reference_checks" / "UPSTREAM_LFP.json")
    parser.add_argument("--devices", "--device", dest="devices", nargs="+", default=["cpu"])
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report, exit_code = run(args)
    report["warnings"] = sorted({f"{item.category.__name__}: {item.message}" for item in captured})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "completed_checks", "passed_checks", "failed_checks", "exit_code")},
                     ensure_ascii=False))
    print(f"Report: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
