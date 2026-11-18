"""Archive only official primary-source DEIM code with pinned Git blob checks."""
import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

OUT = Path(__file__).resolve().parent
REPO = "Intellindust-AI-Lab/DEIM"
API = "https://api.github.com/repos/" + REPO


def fetch(url):
    # curl's TLS backend also works on hosts where urllib's handshake fails.
    result = subprocess.run(["curl", "-4", "--retry", "8", "--retry-all-errors", "--retry-delay", "1", "--connect-timeout", "10",
                             "--max-time", "20", "-fsSL", url], capture_output=True)
    if result.returncode == 0:
        return result.stdout
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepPro-local-reference-audit"})
            return urllib.request.urlopen(request, timeout=15).read()
        except Exception as error:
            last_error = error
    raise last_error


def main():
    commit_metadata = OUT / "DEIM_COMMIT_API.json"
    if not commit_metadata.exists():
        commit_metadata.write_bytes(fetch(API + "/commits/main"))
    commit = json.loads(commit_metadata.read_bytes())["sha"]
    tree_metadata = OUT / "DEIM_TREE_API.json"
    if not tree_metadata.exists():
        tree_metadata.write_bytes(fetch(API + "/git/trees/" + commit + "?recursive=1"))
    tree = json.loads(tree_metadata.read_bytes())["tree"]
    requested = {"engine/deim/deim_criterion.py": "DEIM_deim_criterion.py",
                 "README.md": "DEIM_README.md", "LICENSE": "DEIM_LICENSE"}

    def download(pair):
        source, local = pair
        node = next(n for n in tree if n["path"] == source)
        blob = json.loads(fetch(API + "/git/blobs/" + node["sha"]))
        assert blob["encoding"] == "base64"
        content = base64.b64decode(blob["content"])
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        assert digest == node["sha"]
        (OUT / local).write_bytes(content)
        return {"source_path": source, "local_file": local, "git_blob_sha1": digest,
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob_verified": True,
                "pinned_url": "https://github.com/" + REPO + "/blob/" + commit + "/" + source}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        files = list(executor.map(download, requested.items()))
    code = (OUT / "DEIM_deim_criterion.py").read_text()
    assert "def loss_labels_mal(" in code
    assert "src_logits = outputs['pred_logits']" in code
    assert "F.binary_cross_entropy_with_logits(src_logits, target_score" in code
    start = next(i for i, line in enumerate(code.splitlines(), 1) if "def loss_labels_mal(" in line)
    record = {"name": "DEIM: DETR with Improved Matching for Fast Convergence",
        "venue": "CVPR 2025", "pages": "15162-15171",
        "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "official_proceedings": "https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html",
        "official_project": "https://www.shihuahuang.cn/DEIM/",
        "official_repository": "https://github.com/" + REPO,
        "provenance_chain": "CVF proceedings link author project; author project GitHub link identifies this repository.",
        "commit": commit, "files": files, "mal_function_line": start,
        "verified_code_behavior": "MAL uses actual pred_logits in BCE-with-logits; matched-box IoU is detached, exponentiated by gamma into positive target scores; detached prediction scores weight unmatched negatives.",
        "borrowed_principle": "Connect an auxiliary objective directly to actual detection readouts and quality.",
        "not_borrowed": ["DETR matcher", "IoU quality labels", "Dense O2O augmentation", "DEIM model or parameters", "MAL implementation"],
        "experiment_distinction": "The continuous-segment soft-OR surrogate is the user's C-v2 proposal, not a formula or validated result of DEIM. The DEIM snapshot is evidence only and is not imported by training.",
        "other_references": "SlotContrast and TrackMAE retain the separately verified pinned records under track_contrast_v1/resources; no historical file is modified."}
    (OUT / "DEIM_REFERENCE.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"commit": commit, "mal_function_line": start, "files": files}), flush=True)


if __name__ == "__main__":
    main()
