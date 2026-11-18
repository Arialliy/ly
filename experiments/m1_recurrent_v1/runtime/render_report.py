"""Render the fixed2048 M1 endpoint from completed, hashed CSV evidence only.

This reporting utility is outside the immutable model/training source tree.
Fixture mode can only write below checks/evaluation_fixture_only and prominently
labels every artifact TEST_FIXTURE. It never supplies placeholder metric values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

E = Path(__file__).resolve().parents[1]
CONDITIONS = ("release", "Current", "History")
PLANS = ("O", "S")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scalar(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{key: scalar(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def one(rows, **selectors):
    found = [row for row in rows if all(row.get(key) == value for key, value in selectors.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {selectors}; got {len(found)}")
    return found[0]


def table(headers, rows):
    def display(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(display(value) for value in row) + " |" for row in rows)])


def validate_training():
    pair_path = E / "checks/PAIRED_TRAINING.json"
    pair = read_json(pair_path)
    assert pair["status"] == "PASS" and pair["steps_each"] == 2048
    assert pair["identical_initialization"] and pair["identical_ordered_exposure"] and pair["original_unchanged_both"]
    freeze = read_json(E / "PROTOCOL_FREEZE.json")
    runs = {}; evidence = {str(pair_path): sha(pair_path), str(E / "PROTOCOL_FREEZE.json"): sha(E / "PROTOCOL_FREEZE.json")}
    for condition in ("Current", "History"):
        path = E / "training" / condition / "run/RUN.json"
        run = read_json(path)
        assert run["status"] == "complete" and run["final_step"] == run["main_step"] == 2048
        assert run["condition"] == condition and run["original_unchanged"]
        assert run["initial_model_state_sha256"] == pair["initial_model_state_sha256"]
        assert run["exposure_digest_sha256"] == pair["exposure_digest_sha256"]
        assert run["source_sha256"] == freeze["source_sha256"]
        assert run["protocol_sha256"] == sha(E / "PROTOCOL_FREEZE.json")
        assert sha(path) == pair["run_sha256"][condition]
        endpoint = one(run["snapshots"], step=2048)
        snapshot = path.parent / endpoint["path"]
        assert sha(snapshot) == endpoint["sha256"]
        evidence[str(path)] = sha(path); evidence[str(snapshot)] = sha(snapshot)
        runs[condition] = run
    for key in ("settings", "model_settings", "optimizer", "trainable_parameter_names",
                "original_state_sha256", "initial_checkpoint_sha256", "seed"):
        assert runs["Current"][key] == runs["History"][key], key
    resources = []
    for condition, run in runs.items():
        inference = []
        for sequence in ("DJI_0026_2", "DJI_0067_2"):
            path = E / "predictions" / condition / "step_2048" / sequence / "COMPLETE.json"
            record = read_json(path)
            assert record["status"] == "COMPLETE" and record["checkpoint_step"] == 2048 and record["checkpoint_is_main"]
            assert record["training_run_sha256"] == sha(E / "training" / condition / "run/RUN.json")
            assert record["source_sha256"] == run["source_sha256"]
            assert record["original_unchanged"] and record["all_model_state_unchanged_during_inference"]
            inference.append(record); evidence[str(path)] = sha(path)
        resources.append({"condition": condition, "parameters": run["parameters"],
                          "trainable_parameters": run["trainable_parameters"],
                          "training_seconds": run["elapsed_seconds"],
                          "mean_training_step_seconds": run["elapsed_seconds"] / 2048,
                          "training_peak_allocated_bytes": run["max_cuda_memory_allocated_bytes"],
                          "O_inference_seconds_two_clips": sum(row["O_model_seconds"] for row in inference),
                          "S_inference_seconds_two_clips": sum(row["S_model_seconds"] for row in inference),
                          "union_inference_seconds_two_clips": sum(row["union_model_seconds"] for row in inference),
                          "inference_peak_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in inference)})
    return resources, evidence


def build_report(evaluation, fixture):
    complete_path = evaluation / "SUMMARY_COMPLETE.json"
    complete = read_json(complete_path)
    assert complete["status"] == "COMPLETE" and complete["threshold"] == .5
    assert complete["fixed_main_step"] == 2048 and complete["conditions"] == list(CONDITIONS)
    if not fixture:
        assert complete["evaluated_step"] == 2048 and complete["checkpoint_is_main"]
    required = ("SUMMARY", "UNIQUE_RECOVERY", "PROTECTED_AND_WINDOW_RETENTION", "ALL_GT_SUMMARY",
                "CONTINUOUS_MISS_SUMMARY", "COVERAGE_AND_ISOLATION_SUMMARY", "RECOVERY_CONCENTRATION",
                "TRACK_COVERAGE", "CONTINUOUS_MISS_TRACKS", "RELEASE_MISS_SEGMENTS")
    evidence = {str(complete_path): sha(complete_path)}
    tables = {}
    for name in required:
        path = evaluation / f"{name}.csv"
        assert sha(path) == complete["output_sha256"][path.name], path
        evidence[str(path)] = sha(path); tables[name] = read_csv(path)
    resources, training_evidence = ([], {}) if fixture else validate_training()
    evidence.update(training_evidence)
    main = []; background = []; all_gt = []; continuous = []; coverage = []; protection = []
    for condition in CONDITIONS:
        protected = one(tables["PROTECTED_AND_WINDOW_RETENTION"], condition=condition, cohort="both_fixed_success")
        assert protected["target_frames"] == 646
        assert protected["both_matched"] + protected["lost_dual_success"] == 646
        protection.append(protected)
        for plan in PLANS:
            row = one(tables["SUMMARY"], condition=condition, arrangement=plan, group="pooled")
            assert row["small_gt"] == 5200 and row["low_all_05_gt"] == 4342 and row["extreme_all_001_gt"] == 4170
            assert row["U_budget"] == 4015 and row["threshold"] == .5
            assert row["passes_pooled_U_budget"] == int(row["unmatched_components"] <= 4015)
            assert row["M"] - row["release_M_same"] == row["G_vs_release_same"] - row["L_vs_release_same"]
            assert row["M"] - row["Current_M_same"] == row["G_vs_Current_same"] - row["L_vs_Current_same"]
            main.append({key: row[key] for key in ("condition", "arrangement", "M", "unmatched_components",
                "G_vs_release_same", "L_vs_release_same", "net_vs_release_same", "G_vs_Current_same", "L_vs_Current_same",
                "net_vs_Current_same", "low_all_05_matched", "extreme_all_001_matched", "passes_pooled_U_budget")})
            background.append({key: row[key] for key in ("condition", "arrangement", "unmatched_components",
                "delta_unmatched_components_vs_release", "delta_unmatched_components_vs_Current",
                "unmatched_component_area_sum", "delta_unmatched_component_area_sum_vs_release",
                "delta_unmatched_component_area_sum_vs_Current", "positive_pixels_outside_boxes",
                "delta_positive_pixels_outside_boxes_vs_release", "delta_positive_pixels_outside_boxes_vs_Current",
                "unmatched_component_area_max", "unmatched_component_area_q95")})
            continuous.append(one(tables["CONTINUOUS_MISS_SUMMARY"], condition=condition, arrangement=plan, group="pooled", cohort="all"))
            coverage.append(one(tables["COVERAGE_AND_ISOLATION_SUMMARY"], condition=condition, arrangement=plan, group="pooled"))
            for scope in ("all_valid_gt", "other_valid_gt"):
                all_gt.append(one(tables["ALL_GT_SUMMARY"], condition=condition, arrangement=plan, group="pooled", scope=scope))
    unique = tables["UNIQUE_RECOVERY"]
    for condition, reference in (("Current", "release"), ("History", "release"), ("History", "Current")):
        for cohort in ("all", "low_all_05", "extreme_all_001", "both_fixed_success"):
            row = one(unique, condition=condition, reference=reference, cohort=cohort)
            assert row["gains_O"] + row["gains_S"] == row["unique_gains_either_arrangement"] + row["gains_both_arrangements"]
    concentration = [row for row in tables["RECOVERY_CONCENTRATION"]
                     if row["condition"] == "History" and row["reference"] == "Current" and row["outcome"] == "gain"]
    assert len(concentration) == 2
    persistent_segments = []
    undiscovered_tracks = []
    for plan in PLANS:
        segments = [row for row in tables["RELEASE_MISS_SEGMENTS"]
                    if row["arrangement"] == plan and row["target_frames"] >= 2]
        persistent_segments.append({"arrangement": plan, "release_miss_segments_ge2": len(segments),
            "release_missed_frames_in_segments_ge2": sum(row["target_frames"] for row in segments),
            "Current_recovered_frames": sum(row["Current_recovered_frames"] for row in segments),
            "History_recovered_frames": sum(row["History_recovered_frames"] for row in segments),
            "History_segments_with_more_recovered_than_Current": sum(row["History_recovered_frames"] > row["Current_recovered_frames"] for row in segments),
            "History_segments_with_fewer_recovered_than_Current": sum(row["History_recovered_frames"] < row["Current_recovered_frames"] for row in segments),
            "History_remaining_longest_shorter_than_Current": sum(row["History_remaining_miss_longest"] < row["Current_remaining_miss_longest"] for row in segments),
            "History_remaining_longest_longer_than_Current": sum(row["History_remaining_miss_longest"] > row["Current_remaining_miss_longest"] for row in segments)})
        never = []
        for row in tables["TRACK_COVERAGE"]:
            if row["condition"] != "release" or row["arrangement"] != plan or row["matched_frames"] != 0:
                continue
            observation = one(tables["CONTINUOUS_MISS_TRACKS"], condition="release", arrangement=plan,
                              sequence=row["sequence"], track_id=row["track_id"], cohort="all")
            if observation["observation_longest"] >= 2:
                never.append(row)
        for condition in CONDITIONS:
            selected = [one(tables["TRACK_COVERAGE"], condition=condition, arrangement=plan,
                            sequence=row["sequence"], track_id=row["track_id"]) for row in never]
            undiscovered_tracks.append({"condition": condition, "arrangement": plan,
                "initially_unmatched_tracks_with_observed_segment_ge2": len(selected),
                "tracks_now_matched": sum(row["matched_frames"] > 0 for row in selected),
                "tracks_now_with_matched_run_ge2": sum(row["longest_matched_run"] >= 2 for row in selected),
                "matched_frames": sum(row["matched_frames"] for row in selected),
                "track_mean_coverage": sum(row["coverage_fraction"] for row in selected) / len(selected) if selected else None})
    sustained = []
    for plan in PLANS:
        current = one(coverage, condition="Current", arrangement=plan, group="pooled")
        history = one(coverage, condition="History", arrangement=plan, group="pooled")
        miss = one(continuous, condition="History", arrangement=plan, group="pooled", cohort="all")
        sustained.append({"arrangement": plan, "net_strict_matches_vs_Current": history["net_vs_Current"],
                          "coverage_track_mean_delta_vs_Current": history["coverage_track_mean"] - current["coverage_track_mean"],
                          "longest_miss_mean_delta_vs_Current": miss["longest_delta_track_mean_vs_Current"],
                          "tracks_shorter_vs_Current": miss["tracks_shorter_vs_Current"],
                          "tracks_longer_vs_Current": miss["tracks_longer_vs_Current"],
                          "tracks_coverage_up_vs_Current": history["tracks_coverage_up_vs_Current"],
                          "tracks_coverage_down_vs_Current": history["tracks_coverage_down_vs_Current"],
                          "isolated010_gains_vs_Current": history["isolated010_gain_vs_Current"],
                          "gains_vs_Current": history["gain_vs_Current"],
                          "multi_frame_run_track_delta_vs_Current": history["tracks_with_multi_frame_run"] - current["tracks_with_multi_frame_run"]})
    h_rows = [row for row in main if row["condition"] == "History"]
    u_pass = all(row["passes_pooled_U_budget"] for row in h_rows)
    consistent_signal = all(row["net_strict_matches_vs_Current"] > 0
                            and row["coverage_track_mean_delta_vs_Current"] > 0
                            and row["longest_miss_mean_delta_vs_Current"] < 0
                            and row["tracks_shorter_vs_Current"] > row["tracks_longer_vs_Current"]
                            for row in sustained)
    no_positive = all(row["net_strict_matches_vs_Current"] <= 0 for row in sustained)
    h_unique = one(unique, condition="History", reference="Current", cohort="low_all_05")
    low_increment = all(h_unique[f"gains_{plan}"] > h_unique[f"losses_{plan}"] for plan in PLANS)
    h_protect = one(protection, condition="History", cohort="both_fixed_success")
    if not u_pass:
        verdict = "History至少一个安排超过共同U≤4015约束，本轮未满足背景代价要求；即使存在恢复，也不能认定M1已经完成职责。"
    elif no_positive:
        verdict = "History在O/S的strict5200净匹配均未优于Current；本轮没有形成支持新增历史状态作用的总体增量证据。"
    elif consistent_signal and low_increment:
        verdict = "O/S均出现History相对Current的匹配、逐轨迹覆盖和连续漏检改善，且低响应集合有去重后的增量恢复，构成历史状态有用的初步信号。仍需结合保护对象损失、背景面积及恢复集中度判断代价，不能据此宣称主要失败已解决。"
    else:
        verdict = "History相对Current的增量尚未在O/S、低响应恢复、逐轨迹覆盖与连续漏检改善之间形成一致证据；总体匹配上涨本身不足以支持历史累积解释。"
    if h_protect["lost_dual_success"]:
        verdict += f" History使646保护集合中{h_protect['lost_dual_success']}个目标帧失去双安排成功，其中{h_protect['lost_both_arrangements']}个两安排均丢失。"
    result = {"status": "TEST_FIXTURE" if fixture else "COMPLETE", "evaluated_step": 2048,
              "evaluation_directory": str(evaluation), "created_unix": time.time(),
              "engineering_vs_research_boundary": "Only endpoint detection evidence determines M1 benefit; unit tests and nonzero gradients are not detection recovery.",
              "main": main, "unique_recovery": unique, "protection": protection, "background": background,
              "all_gt": all_gt, "continuous_miss": continuous, "coverage": coverage,
              "history_vs_current_sustained": sustained, "history_vs_current_gain_concentration": concentration,
              "release_persistent_miss_segments_ge2": persistent_segments,
              "initially_undiscovered_observed_tracks": undiscovered_tracks,
              "resources": resources, "decision": verdict,
              "decision_is_conservative_readout_not_new_preregistered_significance_test": True,
              "history_U_budget_passes_both_arrangements": u_pass,
              "consistent_directional_sustained_signal_both_arrangements": consistent_signal,
              "evidence_sha256": evidence}
    return result


def markdown(result):
    out = []
    if result["status"] == "TEST_FIXTURE":
        out.append("**TEST_FIXTURE：以下仅为渲染测试，输入为发布结果复制夹具，不是M1训练结果。**")
    out.append("M1固定2048步结果。" + result["decision"])
    out.append("比较均为seed46、相同初始化与曝光、原DeepPro-Plus参数及BN冻结、同一F监督；History使用上一状态，Current以当前特征替代历史输入并执行相同递归单元。O/S是同一目标的两种窗口安排。")
    out.append("**主结果：strict5200实际匹配M与未匹配组件U**")
    out.append(table(["条件", "安排", "M", "U", "相对发布G/L", "相对Current G/L", "低响应4342匹配", "极低4170匹配", "U≤4015"],
        [[r["condition"], r["arrangement"], r["M"], r["unmatched_components"],
          f"{r['G_vs_release_same']}/{r['L_vs_release_same']}", f"{r['G_vs_Current_same']}/{r['L_vs_Current_same']}",
          r["low_all_05_matched"], r["extreme_all_001_matched"], bool(r["passes_pooled_U_budget"])] for r in result["main"]]))
    out.append("**按目标帧去重的恢复与丢失**")
    out.append(table(["比较", "集合", "O新增", "S新增", "两安排任一新增（去重）", "两安排均新增", "O丢失", "S丢失", "任一丢失（去重）"],
        [[f"{r['condition']} vs {r['reference']}", r["cohort"], r["gains_O"], r["gains_S"],
          r["unique_gains_either_arrangement"], r["gains_both_arrangements"], r["losses_O"], r["losses_S"],
          r["unique_losses_either_arrangement"]] for r in result["unique_recovery"]]))
    out.append("去重键为(sequence, frame, annotation_id)，不能将O/S相加当作独立恢复。‘任一新增’与‘任一丢失’可能在同一目标的不同安排上同时成立，二者相减不是联合系统的净检出数。")
    out.append("**646保护集合与背景面积**")
    out.append(table(["条件", "O保留", "S保留", "双安排保留", "失去双安排成功", "双安排均丢失"],
        [[r["condition"], r["O_matched"], r["S_matched"], r["both_matched"], r["lost_dual_success"], r["lost_both_arrangements"]] for r in result["protection"]]))
    out.append(table(["条件", "安排", "未匹配组件总面积", "面积Δ发布", "面积ΔCurrent", "标注框外预测像素", "框外Δ发布", "框外ΔCurrent", "最大未匹配面积", "未匹配面积q95"],
        [[r[key] for key in ("condition", "arrangement", "unmatched_component_area_sum", "delta_unmatched_component_area_sum_vs_release",
          "delta_unmatched_component_area_sum_vs_Current", "positive_pixels_outside_boxes", "delta_positive_pixels_outside_boxes_vs_release",
          "delta_positive_pixels_outside_boxes_vs_Current", "unmatched_component_area_max", "unmatched_component_area_q95")] for r in result["background"]]))
    out.append("**连续漏检和轨迹覆盖（完整strict5200观察区间）**")
    out.append(table(["条件", "安排", "漏检帧", "逐轨迹最长漏检均值", "中位数", "q90", "最大值", "相对Current缩短/变长轨迹"],
        [[r["condition"], r["arrangement"], r["missed_frames"], r["longest_miss_track_mean"], r["longest_miss_track_median"],
          r["longest_miss_track_q90"], r["longest_miss_max"], f"{r['tracks_shorter_vs_Current']}/{r['tracks_longer_vs_Current']}"] for r in result["continuous_miss"]]))
    out.append(table(["条件", "安排", "轨迹平均覆盖", "有检出轨迹", "有连续≥2检出轨迹", "孤立010检出", "相对Current新增", "其中孤立010新增", "覆盖提高/下降轨迹"],
        [[r["condition"], r["arrangement"], r["coverage_track_mean"], r["tracks_with_any_match"], r["tracks_with_multi_frame_run"],
          r["isolated010_hits"], r["gain_vs_Current"], r["isolated010_gain_vs_Current"],
          f"{r['tracks_coverage_up_vs_Current']}/{r['tracks_coverage_down_vs_Current']}"] for r in result["coverage"]]))
    out.append(table(["History vs Current", "新增帧", "涉及轨迹", "涉及视频", "最多一轨迹新增帧", "前五轨迹新增帧"],
        [[r["arrangement"], r["frames"], r["tracks"], r["clips"], r["top1_frames"], r["top5_frames"]] for r in result["history_vs_current_gain_concentration"]]))
    out.append(table(["安排", "原连续漏检≥2段数", "这些段的原漏检帧", "Current恢复帧", "History恢复帧", "History恢复更多/更少段数", "History剩余最长漏检更短/更长段数"],
        [[r["arrangement"], r["release_miss_segments_ge2"], r["release_missed_frames_in_segments_ge2"],
          r["Current_recovered_frames"], r["History_recovered_frames"],
          f"{r['History_segments_with_more_recovered_than_Current']}/{r['History_segments_with_fewer_recovered_than_Current']}",
          f"{r['History_remaining_longest_shorter_than_Current']}/{r['History_remaining_longest_longer_than_Current']}"] for r in result["release_persistent_miss_segments_ge2"]]))
    out.append(table(["条件", "安排", "原始终未匹配的观察轨迹", "现有匹配轨迹", "现有连续≥2匹配轨迹", "实际匹配帧", "逐轨迹平均覆盖"],
        [[r["condition"], r["arrangement"], r["initially_unmatched_tracks_with_observed_segment_ge2"],
          r["tracks_now_matched"], r["tracks_now_with_matched_run_ge2"], r["matched_frames"], r["track_mean_coverage"]] for r in result["initially_undiscovered_observed_tracks"]]))
    out.append("‘原始终未匹配’仅指固定strict观察集合中发布模型零匹配、且至少有一段连续≥2帧观测的轨迹，不表示完整物理轨迹从头到尾漏检。这些子集按发布结果定义，仅作描述性诊断。")
    out.append("缺标、帧断点及固定集合选择产生的空隙均切断连续段；这些是有标注的观察区间代理，不是已核验的完整物理轨迹或可见性区间。孤立010只统计两侧相邻观测都存在的中心帧；边界检出不自动算持续恢复。最长漏检缩短也须结合覆盖和新增检出分布。")
    out.append("**全部有效标注的匹配影响**")
    out.append(table(["条件", "安排", "范围", "目标帧", "匹配", "相对发布G/L", "相对Current G/L"],
        [[r["condition"], r["arrangement"], r["scope"], r["target_frames"], r["M"],
          f"{r['G_vs_release']}/{r['L_vs_release']}", f"{r['G_vs_Current']}/{r['L_vs_Current']}"] for r in result["all_gt"]]))
    out.append("M只计strict5200；U由全部有效GT匹配后的未匹配组件得到，因此不能用组件总数减M重建U。M/U是诊断量，不是论文Pd/Fa。")
    if result["resources"]:
        out.append("**实际计算成本**")
        out.append(table(["条件", "总/可训练参数", "2048训练秒", "平均每步秒", "训练峰值GiB", "O推理秒（两视频）", "S推理秒（两视频）", "去重窗口推理总秒", "推理峰值GiB"],
            [[r["condition"], f"{r['parameters']}/{r['trainable_parameters']}", r["training_seconds"],
              r["mean_training_step_seconds"],
              r["training_peak_allocated_bytes"] / 2**30, r["O_inference_seconds_two_clips"], r["S_inference_seconds_two_clips"],
              r["union_inference_seconds_two_clips"], r["inference_peak_allocated_bytes"] / 2**30] for r in result["resources"]]))
        out.append("O/S推理时间共享部分窗口，不能相加当作实际总时间；推理时间包含模型和结果复制至CPU。训练使用梯度重计算；冻结前端分块，M1保持全图执行，原逐点预测头按时间分块。官方DCN原子操作及池化反传不保证位级确定性。")
    else:
        out.append("TEST_FIXTURE未读取正式训练和推理日志，计算成本不填写。")
    out.append("仅为单seed、两开发视频的固定终点筛选。确认集未用于本轮；本报告不将工程测试、梯度或总体涨点替代持续漏检恢复证据，也不安排下一模块或追加训练。")
    out.append(f"证据目录：{result['evaluation_directory']}。输入文件SHA256登记于RESULTS.json。")
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=E / "evaluation/step_2048")
    parser.add_argument("--output", type=Path, default=E)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    evaluation, output = args.evaluation_dir.resolve(), args.output.resolve()
    fixture_root = (E / "checks/evaluation_fixture_only").resolve()
    if args.fixture:
        assert evaluation.is_relative_to(fixture_root) and output.is_relative_to(fixture_root)
    else:
        assert evaluation == (E / "evaluation/step_2048").resolve(), "Formal report requires actual endpoint directory"
        assert not output.is_relative_to(fixture_root)
    result = build_report(evaluation, args.fixture)
    text = markdown(result)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "RESULTS.json", output / "RESULTS.md"]
    assert not any(path.exists() for path in paths), "Do not overwrite an existing report"
    paths[0].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths[1].write_text(text, encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
