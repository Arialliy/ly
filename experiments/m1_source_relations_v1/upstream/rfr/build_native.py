"""Build only this experiment's CUDA extension, without occupying a GPU."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


IMAGE = "sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92"


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    relative = directory.relative_to(root)
    devices = subprocess.check_output([
        "nvidia-smi", "--id=1,2,3", "--query-gpu=index,compute_cap", "--format=csv,noheader,nounits",
    ], text=True)
    architectures = sorted({line.split(",")[1].strip() for line in devices.splitlines() if line.strip()})
    if not architectures:
        raise RuntimeError("Could not read the target architecture from authorized GPUs 1-3.")
    architecture = ";".join(architectures)
    command = [
        "docker", "run", "--rm", "--cpus=2", "--memory=8g", "--memory-swap=8g", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "MAX_JOBS=2", "--env", "OMP_NUM_THREADS=2",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", f"TORCH_CUDA_ARCH_LIST={architecture}",
        "--mount", f"type=bind,src={root},dst=/workspace",
        "--workdir", f"/workspace/{relative}/dcn", "--entrypoint", "python",
        IMAGE, "setup.py", "build_ext", "--inplace",
    ]
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "image": IMAGE, "target_devices": devices.strip().splitlines(),
        "torch_cuda_arch_list": architecture, "cpus": 2, "memory_limit_gb": 8,
        "gpu_devices_exposed": False, "network": "none", "max_jobs": 2,
        "command": [item.replace(str(root), "<project_root>") for item in command],
    }
    report_path = directory / "BUILD_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with (directory / "build.log").open("w") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    report["exit_code"] = completed.returncode
    report["status"] = "compiled" if completed.returncode == 0 else "failed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["binaries"] = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((directory / "dcn").glob("_rfr_dcn*.so"))
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
