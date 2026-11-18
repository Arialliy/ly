"""Read-only fixed-training-set endpoint comparison; no model or dev access."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

E = Path(__file__).resolve().parents[1]
CONDITIONS = ("CONT", "FRAME", "GAP")
READOUT_KEYS = (
    "target_logit_mean", "background_logit_mean", "logit_margin_mean",
    "probability_margin_mean", "evidence_mean",
    "target_logit_above_zero_fraction", "background_logit_above_zero_fraction",
)
CONTRAST_KEYS = ("positive_cosine_mean", "negative_cosine_mean", "cosine_margin_mean")
LOSS_KEYS = ("det", "keep", "C", "base", "frame", "gap")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, indices, condition, coefficient):
    assert [row["index"] for row in rows] == indices
    exposure = [row["readout"]["observations_with_multiplicity"] for row in rows]
    pair_weights = [row["contrast"]["pairs_requested"] for row in rows]
    assert all(n > 0 for n in exposure) and sum(pair_weights) > 0
    # This equality holds on the fixed calibration set. Otherwise a requested
    # pair weighted mean of means-over-used-pairs needs a different definition.
    assert all(row["contrast"]["pairs_used"] == row["contrast"]["pairs_requested"] for row in rows)
    readout = {key: sum(row["readout"][key]*n for row,n in zip(rows,exposure))/sum(exposure)
               for key in READOUT_KEYS}
    contrast = {key: sum(row["contrast"][key]*n for row,n in zip(rows,pair_weights))/sum(pair_weights)
                for key in CONTRAST_KEYS}
    losses = {key: statistics.mean(row["losses"][key] for row in rows) for key in LOSS_KEYS}
    assert all(math.isfinite(v) for values in (readout,contrast,losses) for v in values.values())
    expected_margin = readout["target_logit_mean"]-readout["background_logit_mean"]
    # Each batch's reported margins originate in float32 tensor reductions.
    assert abs(readout["logit_margin_mean"]-expected_margin) < 3e-6
    assert abs(contrast["cosine_margin_mean"]-(contrast["positive_cosine_mean"]-contrast["negative_cosine_mean"])) < 1e-7
    optimized = losses["base"]
    if condition in ("FRAME", "GAP"):
        optimized += coefficient*losses[condition.lower()]
    return {"entry_indices": indices, "entry_count": len(rows),
            "observations_with_multiplicity": sum(exposure), "requested_contrast_pairs": sum(pair_weights),
            "readout_exposure_weights": exposure, "contrast_pair_weights": pair_weights,
            "readout": readout, "contrast": contrast, "losses_eight_batch_mean": losses,
            "optimized_objective_eight_batch_mean": optimized}


def difference(after, before):
    return {name: {key: after[name][key]-before[name][key] for key in before[name]}
            for name in ("readout", "contrast", "losses_eight_batch_mean")}


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"]*len(header)) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    calibration_path = E / "CALIBRATION.json"
    calibration = read(calibration_path)
    assert calibration["status"] == "complete" and calibration["no_parameter_updates"]
    indices = calibration["calibration_entry_indices"]
    assert len(indices) == len(set(indices)) == 8
    coefficient = calibration["fixed_settings"]["lambda_readout"]
    snapshots = {"C_init": summarize(calibration["measurements"],indices,"C_init",coefficient)}
    raw_rows = {"C_init": calibration["measurements"]}
    paths = [calibration_path]
    initial_states = set()
    for condition in CONDITIONS:
        folder = E / "training" / condition / "run"
        run_path, diagnostics_path = folder/"RUN.json", folder/"TRAIN_DIAGNOSTICS.json"
        run, diagnostics = read(run_path), read(diagnostics_path)
        assert run["status"] == "complete" and run["final_step"] == 2304 and run["additional_steps"] == 256
        assert run["frozen_frontend_unchanged"] and run["bn_unchanged"] and run["initial_optimizer_restored"]
        assert run["initial_checkpoint_sha256"] == calibration["initial_checkpoint_sha256"]
        assert run["initial_optimizer_sha256"] == calibration["initial_optimizer_sha256"]
        assert diagnostics["indices"] == indices and not diagnostics["uses_development"]
        initial_states.add(run["initial_state_sha256"])
        snapshots[condition] = summarize(diagnostics["rows"],indices,condition,coefficient)
        raw_rows[condition] = diagnostics["rows"]
        for key in ("readout_exposure_weights", "contrast_pair_weights"):
            assert snapshots[condition][key] == snapshots["C_init"][key]
        paths.extend([run_path, diagnostics_path])
    assert len(initial_states) == 1
    comparisons = [("CONT","C_init"),("FRAME","C_init"),("GAP","C_init"),
                   ("FRAME","CONT"),("GAP","CONT"),("GAP","FRAME")]
    delta = {after+"_minus_"+before: difference(snapshots[after],snapshots[before]) for after,before in comparisons}
    directions = {}
    for after,before in (("FRAME","CONT"),("GAP","CONT"),("GAP","FRAME")):
        directions[after+"_minus_"+before] = {
            key: {"positive_entries":[a["index"] for a,b in zip(raw_rows[after],raw_rows[before])
                                      if a["readout"][key]>b["readout"][key]],
                  "negative_entries":[a["index"] for a,b in zip(raw_rows[after],raw_rows[before])
                                      if a["readout"][key]<b["readout"][key]]}
            for key in ("probability_margin_mean","logit_margin_mean")}
    result = {"status":"complete", "generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uses_development":False, "fixed_training_indices":indices,
        "weighting":{"readout":"segment-index multiset exposure weights; overlapping/repeated observations retained",
                     "contrast":"pairs_requested weights; all requested pairs used on every fixed diagnostic entry",
                     "loss":"unweighted mean of the same8 fixed batch losses, preserving each loss's registered internal reduction",
                     "comparison":"all endpoint changes reported against C_init; FRAME and GAP separately compared with CONT and with each other"},
        "lambda_readout":coefficient,"snapshots":snapshots,"comparisons":delta,
        "paired_batch_directions":directions,
        "limitations":["Training diagnostics are not actual-component detections or held-out evaluation.",
            "ROI-mean logit thresholds and sigmoid evidence are training proxies, not calibrated success probabilities.",
            "Old contrast and new segment readouts use different target subsets; their summary changes cannot by themselves establish a causal mismatch.",
            "Repeated segment exposure counts are not independent observation or sample counts.",
            "C_init versus endpoint includes additional training; FRAME/GAP versus CONT isolates the registered added loss within this fixed single-seed256-step revision.",
            "The three optimized-objective values use different added terms; compare common loss components or the explicit paired deltas for interpretation."],
        "source_sha256":{str(path.relative_to(E)):sha(path) for path in paths}}
    output = E / "resources/TRAINING_CHANGE_SUMMARY.json"
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    fmt = lambda value: format(value,".7g")
    names = ["C_init",*CONDITIONS]
    lines = ["# 固定训练侧读出变化", "",
        "三组均从同一 C2048 权重和 Adam 状态完成256次更新。以下只比较同8个固定训练批次；目标/背景读出按段内重复曝光加权，旧对比按请求配对数加权，各损失保留内部归约后对8批等权平均。没有读取开发预测。", "",
        f"每次诊断含 {snapshots['C_init']['observations_with_multiplicity']:,} 次段观察曝光、{snapshots['C_init']['requested_contrast_pairs']:,} 个旧对比请求配对；曝光权重逐批一致。", "",
        table(["状态","目标logit","背景logit","logit间隔","sigmoid分数间隔","联合证据r"],
              [[name,*[fmt(snapshots[name]['readout'][key]) for key in READOUT_KEYS[:5]]]for name in names]), "",
        table(["状态","正余弦","背景余弦","余弦间隔"],
              [[name,*[fmt(snapshots[name]['contrast'][key]) for key in CONTRAST_KEYS]]for name in names]), "",
        table(["状态","det","keep","旧C对比","原C总项","frame","gap"],
              [[name,*[fmt(snapshots[name]['losses_eight_batch_mean'][key]) for key in LOSS_KEYS]]for name in names]), "",
        "## 排除额外训练的增量", "",
        table(["比较","Δ目标logit","Δ背景logit","Δlogit间隔","Δ分数间隔","Δr","Δ旧余弦间隔"],
              [[key,*[fmt(delta[key]['readout'][k]) for k in READOUT_KEYS[:5]],fmt(delta[key]['contrast']['cosine_margin_mean'])]
               for key in ("FRAME_minus_CONT","GAP_minus_CONT","GAP_minus_FRAME")]), "",
        table(["比较","Δdet","Δkeep","Δ旧C对比","Δ原C总项","Δframe","Δgap"],
              [[key,*[fmt(delta[key]['losses_eight_batch_mean'][k]) for k in LOSS_KEYS]]
               for key in ("FRAME_minus_CONT","GAP_minus_CONT","GAP_minus_FRAME")]), "",
        "相对CONT，FRAME和GAP在8个固定批次的sigmoid分数间隔都提高，但logit间隔都下降：目标和背景的平均logit同时抬升，背景logit抬升更多。GAP在这8批的分数间隔均低于FRAME；当前训练侧汇总没有显示连续段相对逐帧控制的额外读出优势。背景原本很低时，logit变化与sigmoid分数变化的尺度不同，因此不能把两种间隔混称为同一改善。", "",
        "以上差值均为前者减后者。C_init到三组的完整差值及逐批权重存于同名JSON。读出改善本身不能当作最终匹配恢复；旧对比和新段读出也不是同一目标子集。只有后续原组件匹配、漏检持续性、保护和完整背景指标能判断检测收益及代价。", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"output":str(output),"comparisons":{k:delta[k]["readout"] for k in ("FRAME_minus_CONT","GAP_minus_CONT","GAP_minus_FRAME")}},ensure_ascii=False))


if __name__ == "__main__":
    main()
