"""Run the S0 gate against the frozen manifest, native release files and production loader."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT.parents[1]))

from experiments.irstd_independent_v1.contracts import CLIP_LENGTH, CENTER_INDEX
from experiments.irstd_independent_v1.data.manifest import build_manifest
from experiments.irstd_independent_v1.checks.audit_pixels import audit_pixels
from experiments.irstd_independent_v1.checks.check_backgrounds import audit_background_overlap
from experiments.irstd_independent_v1.checks.check_loader import verify_loader
from experiments.irstd_independent_v1.checks.check_mat_alignment import verify_mat_alignment


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def repository_commit(root):
    if shutil.which("git"):
        return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root),
                                        "rev-parse", "HEAD"], text=True).strip()
    # The project's pinned image contains Python but no Git executable.
    repo = root / ".git"
    head = (repo / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ")
        loose = repo / ref
        if loose.exists():
            head = loose.read_text().strip()
        else:
            packed = (repo / "packed-refs").read_text().splitlines()
            head = next((line.split()[0] for line in packed if line.endswith(" " + ref)), "")
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head), "Could not resolve repository HEAD")
    return head


def check_time_and_split(root, manifest, config):
    """Enumerate centers a second way: split ordered frame numbers into contiguous runs."""
    spec = config["input"]
    require((spec["clip_length"], spec["center_index"], spec["channel_count"]) ==
            (manifest["clip_length"], manifest["center_index"], 1) == (CLIP_LENGTH, CENTER_INDEX, 1),
            "Conflicting clip dimensions in task, data manifest or public contracts")
    require(config["target"]["mode"] == "mask" and config["task"] == "center_frame_mask"
            and config["target"]["binary_values"] == [0, 255], "Wrong task/GT mode")
    require(config["data"]["dataset"] == manifest["dataset"] and
            config["data"]["split_seed"] == config["data"]["training_seed"] == manifest["seed"] and
            config["data"]["dev_sources_from_official_train"] == manifest["dev_count"],
            "Task protocol and manifest disagree on the release or split")
    require(manifest == build_manifest(root), "Frozen manifest differs from a fresh build")
    halves = defaultdict(set)
    exposure = defaultdict(lambda: {"supervised_centers": 0, "input_slots_per_full_pass": 0,
                                    "distinct_input_frames_per_full_pass": 0})
    for sequence in manifest["sequences"]:
        numbers = sorted(row["number"] for row in sequence["frames"])
        expected = []
        for _, indexed in itertools.groupby(enumerate(numbers), key=lambda pair: pair[1] - pair[0]):
            run = [number for _, number in indexed]
            expected.extend(run[CENTER_INDEX:-CENTER_INDEX])
        require(sequence["centers"] == expected, f'{sequence["id"]} has missing or invalid centers')
        split = sequence["split"]
        require(split in {"train", "dev", "test"} and sequence["official_split"] ==
                ("test" if split == "test" else "train"), "Official split crossing")
        require(all(sequence["id"] not in seen for seen in halves.values()), "Duplicate source ID")
        halves[split].add(sequence["id"])
        totals = exposure[split]
        totals["supervised_centers"] += len(expected)
        totals["input_slots_per_full_pass"] += CLIP_LENGTH * len(expected)
        used = {n for c in expected for n in range(c - CENTER_INDEX, c + CENTER_INDEX + 1)}
        totals["distinct_input_frames_per_full_pass"] += len(used)
    require(halves["dev"] == set(manifest["dev_source_ids"]), "Development IDs disagree")
    return dict(exposure)


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_state(directory, report):
    atomic_text(directory / "S0.json", json.dumps(report, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n")


def format_report(report):
    pixels = report["pixel_audit"]
    rows = []
    for name in ("train", "dev", "test"):
        counts = pixels["splits"][name]
        rows.append(f'| {name} | {counts["sources"]} | {counts["frames"]} | '
                    f'{counts["eligible_centers"]} | {counts["positive_centers"]} | '
                    f'{counts["empty_centers"]} | {counts["components_at_centers"]} |')
    train = report["exposure"]["train"]
    statistics = pixels["train_statistics"]
    width = statistics["component_width_px"]
    shifts = statistics["single_component_pair_centroid_shift_px_per_frame"]
    overlap = statistics["overlapping_single_component_pair_centroid_shift_px_per_frame"]
    examples = report["loader"]
    dev_ids = ", ".join(report["dev_source_ids"])
    evaluation = report["evaluation"]
    backgrounds = report["backgrounds"]
    pair = backgrounds["top_pairs"][0]
    pair_scores = [r["correlation_24"] for r in backgrounds["top_pair_temporal_checks"]]
    return f"""# S0 · NUDT-MIRSDT 数据与标签验收

**{report["status"]}**　使用官方合成视频发布包，验收范围为来源划分、17 帧原图输入和第 9 帧的原始 mask。当前训练步数为 0。

| 划分 | 序列 | 原始帧 | 可输出中心帧 | 非空中心帧 | 空中心帧 | 中心帧连通分量 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### 样本和训练曝光

- 发布列表给出的每个 Mix 条目，通过相同序列和五位帧号定位单帧图像及全监督 mask。按归档成员 SHA-256 核验并用 OpenCV 解码了 {pixels["provenance"]["images_and_masks_checked"]:,} 个图像与标签文件。{report["mat_alignment"]["matching_pairs"]:,} 个 MAT 的最后一层与同号 PNG 逐像素一致，可核对的 {report["mat_alignment"]["matching_history_planes"]:,} 个历史层也与对应旧帧一致。
- 每段序列列出 100 帧，物理帧号 1–100。允许输出 9–92，首尾各 8 帧可以提供上下文。当前定义使用中心帧前后各 8 帧，属于离线视频检测。
- 开发序列从官方训练侧按种子 46 的固定哈希顺序选出，分别为 {dev_ids}。官方列表以外的 {report["unassigned_source_count"]} 段保留在原目录。
- 开发集的 {pixels["splits"]["dev"]["eligible_centers"]} 个中心中有 {pixels["splits"]["dev"]["empty_centers"]} 张空标签。后续用这组数据选择阈值，只能观察含目标画面中的误报。
- 遍历全部训练中心一次可提供 {train["supervised_centers"]:,} 张中心监督，对应 {train["input_slots_per_full_pass"]:,} 个输入帧次，共用 {train["distinct_input_frames_per_full_pass"]:,} 张不同原图。实际优化器看到的样本数要在训练时按采样器另记。
- 训练标签按八连通分量统计宽度，样本数 {width["n"]:,}，最小值 {width["min"]:g}、中位数 {width["median"]:g}、P95 {width["p95"]:g}、最大值 {width["max"]:g} 像素。同一目标跨帧重复计入。
- {statistics["adjacent_center_pairs"]:,} 对相邻训练中心中，{shifts["n"]:,} 对的两张 mask 各有一个分量，其可见质心位移中位数 {shifts["median"]:.2f}、P95 {shifts["p95"]:.2f} 像素/帧。其中 {overlap["n"]:,} 对有前景像素重叠。序列没有逐目标 ID 或速度真值，质心受显隐及形状影响，重叠条件也偏向小位移。这些统计不能确定速度搜索上限。

### 真实 batch 和复核

[抽样原图、中心标签叠加与空标签](S0_samples.png)

- 样本编号为 {", ".join(examples["sample_ids"])}，实际输入维度为 `{examples["batch_shapes"]["frames"]}`，顺序为批次、时间、通道、高、宽。不同原图尺寸右侧及底部补齐，填充区的像素有效位为假，监督 ignore 位为真。
- 正常入口读取 17 张原生单通道 PNG，强度除以 255。同一 clip 全部帧和标签使用原始坐标。推理只取图像，训练或评价单独取第 9 帧 mask。核查覆盖每帧像素、中心标签、批填充与异常输入，逐项结果见 [S0.json](S0.json)。

### 评价口径

- 自定义目标级 FROC 在原图有效区域按分数提取八连通区域极大值，平台按行、列取首点。分数降序，相同分数按行列排序。抑制切比雪夫距离 ≤ {evaluation["suppression_chebyshev_radius_px"]} 像素的后续峰。阈值含等号，同分成组。与 mask 连通分量质心做一对一匹配，欧氏距离 ≤ {evaluation["match_max_distance_px_inclusive"]} 像素，先最大化配对数，再最小化距离和。
- 检出率分母为所有中心帧中的 GT 分量数，FP/frame 分母为全部规定中心帧数，包含空帧和图像边缘。在开发集按 0.5、1、5 FP/frame 各自选预算内最高检出率，平局依次选较少 FP、较高阈值，含零预测操作点。冻结阈值后报告迁移结果。评测集扫阈值另列描述性 FROC，报告实际 TP、FP 和逐序列差异。0.5 FP/frame 表示每两张中心帧平均一次误报。
- NUDT 原生评价按作者的 ShootingRules、测试帧集合和配置复核 Pd、Fa 与 AUC。17 帧无补齐协议只能输出官方 2,000 测试帧中的 1,680 帧，按此范围测得的结果必须注明“17 帧有效中心子集”。自定义峰计数不能替代作者 Fa。后续若使用框中心监督，需要单列定位任务和指标。S0 未运行模型评价。

### 适用范围与复算

- 在 {backgrounds["decoded_frames"]:,} 张解码原图中，跨划分的逐像素重复图像有 {backgrounds["exact_cross_split_duplicate_frames"]} 种。对 {backgrounds["cross_split_pairs"]:,} 对跨划分序列比较首帧的 24×24 缩略图，最高相关的一对是 {pair["left"]}（{pair["left_split"]}）和 {pair["right"]}（{pair["right_split"]}），Pearson r 为 {pair["correlation"]:.4f}。这对序列在五个时间位置的缩略图相关系数范围为 {min(pair_scores):.4f}–{max(pair_scores):.4f}。
- 发布序列目录互斥。发布包缺少成像场景及生成背景的上游 ID，上述相似性使这份划分的物理场景独立性存疑。
- 该测试集在本仓库有历史使用记录。真实场景泛化与最终独立确认仍需另行验证。

在仓库根目录的项目 Python 环境中可以复算本阶段结果。

```bash
python experiments/irstd_independent_v1/data/manifest.py
python experiments/irstd_independent_v1/checks/check_s0.py
```

后续训练代码在仓库根目录可通过 `experiments.irstd_independent_v1.data.dataset` 导入 `MaskClipDataset` 与 `collate_clips`。固定 manifest 的 SHA-256 为 `{report["data_manifest_hash"]}`。代码、配置、图像样例摘要和运行环境见 [机器记录](S0.json)。
"""


def run_gate(root, directory):
    state = {"stage": "S0", "status": "RUNNING", "gate_kind": "engineering",
             "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
             "training_steps": 0, "failure_owner": None, "next_allowed_stage": None,
             "artifacts": []}
    write_state(directory, state)
    step = "input_contract"
    try:
        config_path = EXPERIMENT / "configs/task.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest_path = EXPERIMENT / config["data"]["manifest"]
        manifest = json.loads(manifest_path.read_bytes())
        state.update(code_commit=repository_commit(root), data_manifest_hash=sha256(manifest_path),
            seeds={"split": config["data"]["split_seed"], "training": config["data"]["training_seed"]},
            dev_source_ids=manifest["dev_source_ids"], unassigned_source_count=len(manifest["unassigned_source_ids"]),
            environment={"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                         "container_image": os.environ.get("S0_IMAGE_ID")},
            evaluation=config["evaluation"], implemented_scope=["mask_video_data", "training_and_inference_interfaces",
                                                                "dataset_integrity", "evaluation_protocol"])
        files = [config_path, *EXPERIMENT.glob("*.py"), *EXPERIMENT.glob("data/*.py"),
                 *EXPERIMENT.glob("checks/*.py")]
        state["code_and_config_sha256"] = {str(p.relative_to(EXPERIMENT)): sha256(p) for p in sorted(files)}
        state["exposure"] = check_time_and_split(root, manifest, config)
        state["checks"].append(step)
        step = "archive_hashes_native_pixels_and_labels"
        state["pixel_audit"] = audit_pixels(root, manifest)
        state["checks"].append(step)
        step = "official_mat_to_png_alignment"
        state["mat_alignment"] = verify_mat_alignment(root, manifest)
        state["checks"].append(step)
        step = "source_background_diagnostic"
        state["backgrounds"] = audit_background_overlap(root, manifest)
        require(state["backgrounds"]["exact_cross_split_duplicate_frames"] == 0,
                "Exact native images repeated across data splits")
        state["split_independence"] = {"released_sequence_ids": "verified_disjoint",
                                       "physical_capture_or_background_ids": "unverified"}
        state["checks"].append(step)
        step = "loader_and_fault_injections"
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            figure = Path(staging) / "S0_samples.png"
            state["loader"] = verify_loader(root, manifest, state["pixel_audit"]["train_samples"], figure)
            state["checks"].append(step)
            step = "report"
            state.update(status="PASS", next_allowed_stage="S1",
                         remaining_limits=["synthetic_release", "no_physical_scene_ids", "no_velocity_or_track_ids",
                                           "official_test_previously_used", "17_frame_subset", "no_model_training",
                                           "no_empty_dev_centers", "background_similarity_across_splits"])
            state["artifacts"] = ["reports/S0.json", "reports/S0_data_contract.md", f"reports/{figure.name}",
                                  str(manifest_path.relative_to(directory.parent))]
            state["figure_sha256"] = sha256(figure)
            markdown = format_report(state)
            os.replace(figure, directory / figure.name)
            atomic_text(directory / "S0_data_contract.md", markdown)
    except Exception as error:
        if hasattr(error, "result"):
            state[step] = error.result
        state.update(status="BLOCKED" if isinstance(error, FileNotFoundError) else "FAIL",
                     next_allowed_stage=None, failure_owner=step, artifacts=[],
                     error=f"{type(error).__name__}: {error}")
        atomic_text(directory / "S0_data_contract.md",
                    f'# S0 · {state["status"]}\n\n未通过步骤 `{step}`。\n\n{state["error"]}\n')
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(directory, state)
    print(f'S0 {state["status"]} | {directory / "S0.json"}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=EXPERIMENT.parents[1])
    args = parser.parse_args()
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lock = os.open(reports, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_gate(args.project_root.resolve(), reports)
    finally:
        os.close(lock)
