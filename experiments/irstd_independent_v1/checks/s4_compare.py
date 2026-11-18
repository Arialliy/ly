"""Decide the common S4 training budget and report the paired development comparison."""
import argparse
import csv
import io
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile

import numpy as np

from .s4_lineage import diagnostic_origin, parent_run, runtime_chunks


MODES = ("M0", "Msep", "Mraw", "Mint")
SEEDS = (46,)
BUDGETS = ("0.5", "1", "5")
HYPOTHESES = {"H1": ("Mraw", "Msep"), "H2": ("Mint", "Mraw")}


def _read(path):
    return json.loads(path.read_bytes())


def _sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _write(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Previous and reconstructed records diverge: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".s4-compare-", mode="w",
                                     encoding="utf-8", delete=False) as out:
        tmp = Path(out.name)
        try:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
            os.link(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _save(path, record):
    _write(path, json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _protocol(run_dir):
    p = _read(run_dir / "protocol.json")
    _check(tuple(p["modes"]) == MODES and tuple(p["seeds"]) == SEEDS,
           "Unexpected paired models or seeds")
    for key, size, n_sources in (("dev_ids", 672, 8), ("test_ids", 1680, 20)):
        ids = p[key]
        counts = Counter(name.split("/")[0] for name in ids)
        _check(len(set(ids)) == len(ids) == size and len(counts) == n_sources and set(counts.values()) == {84},
               f"Wrong frozen sources or frame IDs in {key}")
    _check(len(p["fit_centers"]) == 144 and len(p["diagnostic_dev_ids"]) == 32, "Incorrect diagnostic panels")
    return p


def _fit_loss(folder, p, step):
    run, mode = folder.parents[1], folder.parent.name
    source, protocol_sha = diagnostic_origin(run, p, mode, step)
    at = source / mode / "seed46"
    j = _read(at / f"diagnostics_{step}.json")
    side = _read(at / "checkpoints" / f"step_{step:06d}.json")
    _check((j["mode"], j["seed"], j["step"], j["protocol_sha256"]) == (mode, 46, step, protocol_sha)
           and j["checkpoint_sha256"] == side["checkpoint_sha256"]
           and j["diagnostic_sha256"] == p["diagnostics_sha256"]["train_diagnostics.json"],
           f"Diagnostic does not match its original protocol: {at}/{step}")
    def key(row):
        return row["source"], row["center"], tuple(row["core_xyxy"])
    expected = {key(r) for r in p["fit_centers"]}
    rows = j["fit"]["by_core"]
    _check(j["step"] == step and len(rows) == len(expected) == 144 and {key(r) for r in rows} == expected,
           f"Incorrect training diagnostic cores at {folder} step {step}")
    loss = j["fit"]["loss"]
    _check(math.isfinite(loss) and loss >= 0 and all(math.isfinite(r["total"]) for r in rows)
           and _same(loss, mean(r["total"] for r in rows)), "Diagnostic mean or finiteness mismatch")
    return loss


def _trend(first, middle, last):
    reduction = (first-last)/max(first, 1e-12)
    return {"losses": [first,middle,last], "strictly_decreasing": first > middle > last,
            "fractional_reduction": reduction, "trigger": first > middle > last and reduction >= 0.03}


def decide_budget(run_dir):
    """Choose the shared endpoint after verifying all paired 24k training receipts and diagnostics."""
    run_dir = Path(run_dir)
    p = _protocol(run_dir)
    protocol_hash = _sha(run_dir / "protocol.json")
    monitored = {}
    triggered = []
    evidence = {}
    for mode in MODES:
        monitored[mode] = {}
        for seed in p["seeds"]:
            folder = run_dir/mode/f"seed{seed}"
            training = _read(folder/"training_24000.json")
            _check((training["mode"], training["seed"], training["step"], training["protocol_sha256"]) ==
                   (mode, seed, 24000, protocol_hash), "Missing paired 24k finish")
            path = folder/"checkpoints"/"step_024000.pt"
            _check(training["checkpoint_sha256"] == _sha(path), "24k weights differ from training receipt")
            trend = _trend(*[_fit_loss(folder,p,n) for n in (18000,21000,24000)])
            for n in (18000,21000,24000):
                origin, _ = diagnostic_origin(run_dir,p,mode,n)
                d = origin/mode/f"seed{seed}"/f"diagnostics_{n}.json"
                evidence[os.path.relpath(d, run_dir)] = _sha(d)
            monitored[mode][str(seed)] = trend
            if trend["trigger"]:
                triggered.append(f"{mode}/seed{seed}")
    result = {"selected_step":36000 if triggered else 24000, "triggered_by":triggered,
              "diagnostic_steps":[18000,21000,24000], "by_model_seed":monitored, "seeds":p["seeds"],
              "protocol_sha256":protocol_hash, "diagnostic_sha256":evidence}
    _save(run_dir/"budget_decision.json",result)
    return result


def _point(point, ids, budget=None):
    """Independently reconstruct counts and source rates at one global operating threshold."""
    per_frame = point["per_frame"]
    _check(set(per_frame) == ids, "Incomplete or duplicate metric frame IDs")
    sources = {}
    hits = {}
    for name, row in per_frame.items():
        gt, tp, fp = (row[k] for k in ("gt","tp","fp"))
        _check(gt in (0,1) and 0 <= tp <= gt and type(fp) is int and fp >= 0, f"Invalid frame counts: {name}")
        if tp and "match_distance_px" in row:
            _check(0 <= row["match_distance_px"] <= 3, "Matched centroid exceeds the agreed radius")
        if gt:
            hits[name] = bool(tp)
        src = name.split("/")[0]
        counts = sources.setdefault(src,Counter())
        counts.update(frames=1,gt=gt,tp=tp,fp=fp,empty_frames=int(not gt),empty_frame_fp=0 if gt else fp)
    totals = dict(sum(sources.values(), Counter()))
    for key in ("gt","tp","fp","frames","empty_frames","empty_frame_fp"):
        totals.setdefault(key,0)
        if key in point:
            _check(totals[key] == point[key], f"Summed {key} mismatch")
    if "per_sequence" in point:
        _check(set(sources) == set(point["per_sequence"]), "Missing source at global operating point")
        for s, actual in sources.items():
            reference = point["per_sequence"][s]
            for k in ("frames","gt","tp","fp"):
                _check(reference[k] == actual[k], f"Source count differs for {s}/{k}")
    for row in [totals,*sources.values()]:
        row["pd"] = row["tp"]/row["gt"] if row["gt"] else None
        row["fp_per_frame"] = row["fp"]/row["frames"]
    _check(_same(point["pd"],totals["pd"]) and _same(point["fp_per_frame"],totals["fp_per_frame"]),
           "Reported operating rates differ from exact frame totals")
    if budget is not None:
        _check(totals["fp"] <= float(budget)*len(ids), "Descriptive threshold exceeds requested budget")
    return {**totals, "threshold":point["threshold"], "per_source":dict(sources)}, hits


def _curve_check(description, folder, ids):
    path = folder/description["curve_file"]
    _check(_sha(path) == description["curve_sha256"], "FROC digest mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        thresholds, tp, fp = (arrays[k] for k in ("thresholds","tp","fp"))
        _check(set(arrays["frame_ids"].tolist()) == ids and len(thresholds) == len(tp) == len(fp)
               and np.isposinf(thresholds[0]) and np.all(np.diff(thresholds) < 0)
               and tp[0] == fp[0] == 0 and np.all(np.diff(tp) >= 0) and np.all(np.diff(fp) >= 0),
               "Invalid FROC thresholds or counts")
        for budget in BUDGETS:
            point = description["budgets"][budget]
            allowed = fp <= float(budget)*len(ids)
            best = tp[allowed].max()
            fewest = fp[allowed & (tp == best)].min()
            idx = np.flatnonzero((tp==best) & (fp==fewest))[0]
            threshold = None if idx == 0 else float(thresholds[idx])
            _check((int(best),int(fewest),threshold) == (point["tp"],point["fp"],point["threshold"]),
                   "Saved curve disagrees with the exact operating point")


def _check_binding(run_dir, folder, split_folder, p, mode, seed, step, train, ids):
    binding = _read(split_folder/"binding.json")
    for key,value in (("mode",mode),("seed",seed),("step",step),("protocol_sha256",_sha(run_dir/"protocol.json")),
                      ("checkpoint_sha256",train["checkpoint_sha256"])):
        _check(binding[key] == train[key] == value, f"Endpoint dependency mismatch: {split_folder}/{key}")
    _check((binding["query_chunk"], binding["velocity_chunk"]) == runtime_chunks(p,mode)
           and binding["execution_binding_sha256"] == _sha(run_dir/"execution_binding.json"),
           "Endpoint evaluation used a different arithmetic contract")
    _check(binding["metrics_sha256"] == _sha(split_folder/"metrics.json"), "Final metrics digest mismatch")
    _check(set(binding["frame_sha256"]) == ids, "Raw output population disagrees with protocol")
    for name, checksum in binding["frame_sha256"].items():
        _check(_sha(split_folder/"logits"/(name.replace("/", "_")+".npy")) == checksum, f"Changed output: {name}")
    return binding


def _verify_native(metrics):
    if metrics is None or "overall" not in metrics:
        return {"status":"pending", "reason":metrics}
    def compact(r):
        keys = ("frames","pixels","gt","tp_at_probability_0.5","false_pixels_at_probability_0.5",
                "pd_at_probability_0.5","fa_at_probability_0.5","auc","foreground_iou_at_probability_0.5")
        out = {k:r[k] for k in keys if k in r}
        _check(isinstance(out["frames"],int) and out["frames"] > 0 and out["pixels"] > 0, "Invalid native totals")
        if "false_pixels_at_probability_0.5" in out:
            _check(_same(out["fa_at_probability_0.5"],out["false_pixels_at_probability_0.5"]/out["pixels"]),
                   "Native Fa normalization mismatch")
        return out
    overall = compact(metrics["overall"])
    by_source = {k:compact(v) for k,v in metrics["per_source"].items()}
    for key in ("frames","gt","pixels","tp_at_probability_0.5","false_pixels_at_probability_0.5"):
        if key in overall:
            _check(sum(r[key] for r in by_source.values()) == overall[key], "Native counts do not sum across sources")
    groups = {}
    if "snr_groups" in metrics:
        assigned = []
        for label, group in metrics["snr_groups"].items():
            src = group["sources"]
            compact(group)
            for key in ("frames", "gt", "pixels", "tp_at_probability_0.5", "false_pixels_at_probability_0.5"):
                _check(group[key] == sum(by_source[s][key] for s in src), f"SNR group differs from sources: {label}/{key}")
            assigned.extend(src)
            groups[label] = group
        _check(set(groups) == {"high","low"} and sorted(assigned) == sorted(by_source), "Incorrect SNR source partition")
    return {"status":"complete", "overall":overall, "per_source":by_source, "snr_groups":groups}


def _run_result(run_dir,p,mode,seed,step,lock):
    folder = run_dir/mode/f"seed{seed}"
    train = _read(folder/f"training_{step}.json")
    schedule = _sha(run_dir/f"schedule_{seed}.json")
    _check(train["checkpoint_sha256"] == _sha(folder/"checkpoints"/f"step_{step:06d}.pt")
           and schedule == train["schedule_sha256"] == p["schedule_sha256"][str(seed)], "Changed endpoint state or schedule")
    history = {n:_fit_loss(folder,p,n) for n in range(0,step+1,3000)}
    result = {"step":step, "checkpoint_sha256":train["checkpoint_sha256"], "fit_loss_by_step":history,
              "endpoint_learning_trend":_trend(*[history[step-x] for x in (6000,3000,0)])}
    if step == 36000:
        # The only common intermediate endpoint with a budget decision and a complete paired dev population.
        first = folder / "dev_24000"
        old = _read(folder / "training_24000.json")
        ids = set(p["dev_ids"])
        _check(old["checkpoint_sha256"] == _sha(folder / "checkpoints/step_024000.pt"), "24k state changed")
        _check_binding(run_dir,folder,first,p,mode,seed,24000,old,ids)
        desc = _read(first / "metrics.json")["descriptive"]
        _curve_check(desc,first,ids)
        point, _ = _point(desc["budgets"]["1"],ids,1)
        result["development_24000_at_one_fp_per_frame"] = {
            key:point[key] for key in ("threshold","frames","gt","tp","fp","pd","fp_per_frame")}
    identities = {}
    for split,n_gt,n_empty in (("dev",672,0),("test",1466,214)):
        out = folder/f"{split}_{step}"
        ids = set(p[f"{split}_ids"])
        binding = _check_binding(run_dir,folder,out,p,mode,seed,step,train,ids)
        metrics = _read(out/"metrics.json")
        desc = metrics["descriptive"]
        _check(desc["frames"] == len(ids) and desc["gt"] == n_gt and desc["empty_frames"] == n_empty,
               f"Wrong {split} endpoint labels or denominator")
        _curve_check(desc,out,ids)
        item = {"metrics_sha256":binding["metrics_sha256"], "curve_sha256":desc["curve_sha256"],
                "native":_verify_native(metrics["native_author"])}
        for category in (("descriptive",) if split=="dev" else ("descriptive","transferred")):
            choices = desc["budgets"] if category == "descriptive" else metrics["transferred"]
            item[category] = {}
            for b in BUDGETS:
                counts,hits = _point(choices[b],ids,b if category=="descriptive" else None)
                _check((counts["gt"],counts["empty_frames"]) == (n_gt,n_empty), "Wrong endpoint denominator")
                item[category][b] = counts
                identities[(split,category,b)] = hits
                if category == "transferred":
                    development = result["dev"]["descriptive"][b]["threshold"]
                    _check(counts["threshold"] == development == lock["runs"][mode][str(seed)]["thresholds"][b],
                           "Historical threshold was not frozen on the full development set")
        if split == "dev":
            folds = metrics["leave_one_source_out"]
            _check(set(folds) == set(item["descriptive"]["1"]["per_source"]), "Expected eight removal folds")
            item["leave_one_source_out"] = {}
            for source, fold in folds.items():
                kept = sum(1 for k in ids if k.split("/")[0]!=source)
                row = fold["budgets"]["1"]
                _check(kept == 588 and row["fp"] <= kept and _same(row["pd"],row["tp"]/kept),
                       "Leave-one-source-out denominator mismatch")
                item["leave_one_source_out"][source] = {k:row[k] for k in ("threshold","pd","tp","fp","fp_per_frame")}
        result[split] = item
    return result, identities


def _hypothesis(runs, candidate, control):
    a, b = (runs[name]["46"] for name in (candidate,control))
    x, y = (row["dev"]["descriptive"]["1"] for row in (a,b))
    difference = Fraction(x["tp"]-y["tp"], x["gt"])
    sources = sorted(x["per_source"])
    per_source = {s:100*(x["per_source"][s]["pd"]-y["per_source"][s]["pd"]) for s in sources}
    excluded = {s:100*(a["dev"]["leave_one_source_out"][s]["pd"]-b["dev"]["leave_one_source_out"][s]["pd"])
                for s in sources}
    flips = [s for s,delta in excluded.items() if delta*difference<0]
    positive_lost = [s for s,delta in excluded.items() if difference>0 and delta<=0]
    learning = [name for name in (candidate, control) if runs[name]["46"]["endpoint_learning_trend"]["trigger"]]
    if learning:
        outcome = "optimization_uncertain"
    elif difference >= Fraction(1,100) and (flips or positive_lost):
        outcome = "source_sensitive"
    elif difference >= Fraction(1,100):
        outcome = "single_seed_positive_signal"
    else:
        outcome = "single_seed_below_one_pp"
    return {"candidate":candidate,"control":control,"seed":46,"outcome":outcome,
            "gain_pp":100*float(difference),"gt_gain":x["tp"]-y["tp"],"at_least_one_pp":difference>=Fraction(1,100),
            "per_source_gain_pp":per_source,"leave_one_source_out_gain_pp":excluded,
            "excluded_sources_flipping_sign":flips, "excluded_sources_losing_positive_sign":positive_lost,
            "endpoint_continued_learning":learning,"cross_seed_stability":"unmeasured"}


SCOPES = (("dev","descriptive"),("test","transferred"),("test","descriptive"))


def _comparison_csv(runs):
    stream = io.StringIO(newline="")
    cols = ["split","threshold_rule","mode","seed","budget_fp_per_frame","threshold_raw_logit","frames",
            "empty_frames","gt","tp","fp","pd","fp_per_frame","empty_frame_fp"]
    writer = csv.writer(stream,lineterminator="\n")
    writer.writerow(cols)
    for split, rule in SCOPES:
        for mode in MODES:
            for budget in BUDGETS:
                point = runs[mode]["46"][split][rule][budget]
                writer.writerow([split,rule,mode,46,budget,point["threshold"],
                                 *[point[k] for k in cols[6:]]])
    return stream.getvalue()


def _paired_csv(identities):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream,lineterminator="\n")
    writer.writerow(["split","threshold_rule","budget_fp_per_frame","seed","source_id","center_frame",
                     "gt_component_id","Mint_hit","Mraw_hit","identity_category"])
    summary = {}
    for split, rule in SCOPES:
        summary[f"{split}/{rule}"] = {}
        for budget in BUDGETS:
            key = (split,rule,budget)
            a,b = (identities[mode][key] for mode in ("Mint","Mraw"))
            _check(set(a)==set(b), "Paired GT identities differ between models")
            counts = Counter()
            names = {(True,True):"both", (True,False):"Mint_only", (False,True):"Mraw_only", (False,False):"neither"}
            for frame in sorted(a):
                source, center = frame.split("/")
                category = names[a[frame],b[frame]]
                counts[category] += 1
                writer.writerow([split,rule,budget,46,source,int(center),1,int(a[frame]),int(b[frame]),category])
            summary[f"{split}/{rule}"][budget] = {name:counts[name] for name in names.values()}
    return stream.getvalue(), summary


def _markdown(result):
    outcomes = {"single_seed_positive_signal":"该种子达到 1.0 个百分点门槛",
                "single_seed_below_one_pp":"该种子未达到 1.0 个百分点门槛",
                "source_sensitive":"该种子达到门槛，删去一个来源后正收益消失",
                "optimization_uncertain":"终点训练诊断仍在明显改善，训练充分性存疑"}
    lines = ["# S4 · 固定种子 46 的红外小目标检测比较", "",
             f"四个模型以相同裁块日程训练至 {result['step']:,} 步。主比较在合成发布数据的 8 个开发来源、672 个中心帧上进行。每帧各有一个标注目标。", ""]
    if "migration" in result:
        lineage = result["migration"]
        lines += [f"三个证据模型共同继承父运行 `{lineage['parent_run_id']}` 的前 {lineage['common_step']:,} 次更新，"
                  f"计算分块从 {lineage['parent_chunks']} 改为 {lineage['child_chunks']}，继续执行同一份种子 46 裁块日程。"
                  "M0 从配对初始状态独立训练。父运行原始诊断保留其原始摘要。",""]
    if result["step"] == 36000:
        lines += ["## 延长预算后的开发集结果","",
                  "同一模型均在 672 帧完整开发集和 1 FP/帧预算下取工作点。", "",
                  "| 模型 | 24k TP / FP / Pd | 36k TP / FP / Pd | Pd 变化（百分点） |",
                  "|---|---:|---:|---:|"]
        for mode in MODES:
            r = result["runs"][mode]["46"]
            early, final = r["development_24000_at_one_fp_per_frame"],r["dev"]["descriptive"]["1"]
            points = [f"{v['tp']} / {v['fp']} / {100*v['pd']:.2f}%" for v in (early,final)]
            lines.append("| "+mode+" | "+" | ".join(points)+f" | {100*(final['pd']-early['pd']):+.2f} |")
        lines.append("")
    for name, delta in result["research_outcome"].items():
        lines.extend([f"**{name}（{delta['candidate']} 对 {delta['control']}）**　{outcomes[delta['outcome']]}。"
                      f"每帧至多 1 个虚警时，检出率相差 {delta['gain_pp']:+.2f} 个百分点，"
                      f"逐帧标注检出数相差 {delta['gt_gain']:+d} 个。",""])
    reference = result["reference_gain_pp"]
    lines += [f"作为净收益参照，Mint 相对 M0 的开发集主工作点差为 {reference['Mint-M0']:+.2f} 个百分点，"
              f"相对 Msep 的差为 {reference['Mint-Msep']:+.2f} 个百分点。","",
              "## 全局工作点","",
              "每一格依次为 TP/FP。阈值迁移使用完整开发集冻结的原始 logit 阈值，因而历史测试上的实际虚警/帧可能超出标称预算。历史测试共 1,680 个中心帧，含 214 个空帧。","",
              "| 模型 | 开发 0.5 | 开发 1 | 开发 5 | 历史迁移 0.5 | 历史迁移 1 | 历史迁移 5 | 历史描述 0.5 | 历史描述 1 | 历史描述 5 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        r = result["runs"][mode]["46"]
        cells = [mode]
        for split, rule in SCOPES:
            cells.extend(f"{r[split][rule][b]['tp']}/{r[split][rule][b]['fp']}" for b in BUDGETS)
        lines.append("| "+" | ".join(cells)+" |")
    lines += ["","## 来源敏感性","",
              "单来源列采用全开发集统一阈值。删去该来源列在余下 7 个来源上重新扫描阈值。各格为检出率百分点差。","",
              "| 来源 | H1 单来源 | H1 删去该来源 | H2 单来源 | H2 删去该来源 |","|---|---:|---:|---:|---:|"]
    a,b = (result["research_outcome"][key] for key in ("H1","H2"))
    for source in a["per_source_gain_pp"]:
        vals = [v[k][source] for v in (a,b) for k in ("per_source_gain_pp","leave_one_source_out_gain_pp")]
        lines.append("| "+source+" | "+" | ".join(f"{v:+.2f}" for v in vals)+" |")
    lines += ["","## 作者原生指标与可解释范围","",
              "作者概率阈值 0.5 的 Fa 以像素数归一。报告使用对应序列的原始质心标签，历史测试按 17 帧有效中心子集计算。","",
              "| 集合 | 模型 | 作者 Pd@0.5 | 作者 Fa@0.5 | 作者 AUC |","|---|---|---:|---:|---:|"]
    for split in ("dev","test"):
        for mode in MODES:
            native = result["runs"][mode]["46"][split]["native"]
            if native["status"] == "complete":
                n = native["overall"]
                fields = [f"{n[k]:.6g}" if n.get(k) is not None else "待完成"
                          for k in ("pd_at_probability_0.5","fa_at_probability_0.5","auc")]
            else:
                fields = ["待完成"]*3
            lines.append("| "+("开发" if split=="dev" else "历史测试")+" | "+mode+" | "+" | ".join(fields)+" |")
    lines += ["", "历史测试的信噪比分组沿用作者发布的序列顺序。", "",
              "| 模型 | SNR | 帧数 | 作者 Pd@0.5 | 作者 Fa@0.5 | 作者 AUC |", "|---|---|---:|---:|---:|---:|"]
    for mode in MODES:
        for group, counts in result["runs"][mode]["46"]["test"]["native"].get("snr_groups", {}).items():
            fields = [f"{counts[k]:.6g}" if counts[k] is not None else "无定义"
                      for k in ("pd_at_probability_0.5","fa_at_probability_0.5","auc")]
            lines.append("| "+mode+" | "+group+" | "+str(counts["frames"])+" | "+" | ".join(fields)+" |")
    lines += ["","固定一个种子使配对比较可以复现。跨种子波动和跨新来源泛化仍需另行测量。开发来源互相缺少物理场景独立性验证，历史测试集已有使用记录。",
              "", "[36 个工作点的阈值、计数和虚警率](S4_comparison.csv) · [逐帧配对身份](S4_paired_targets.csv) · [机器汇总](S4_summary.json)",""]
    return "\n".join(lines)


def emit_summary(run_dir, step):
    run_dir = Path(run_dir)
    p = _protocol(run_dir)
    decision = decide_budget(run_dir)
    _check(decision["selected_step"] == step, "Endpoint does not match the common budget decision")
    lock = _read(run_dir/"historical_lock.json")
    runs, identities = {}, {}
    for mode in MODES:
        row, identities[mode] = _run_result(run_dir,p,mode,46,step,lock)
        runs[mode] = {"46":row}
    hypothesis = {key:_hypothesis(runs,a,b) for key,(a,b) in HYPOTHESES.items()}
    points = {m:runs[m]["46"]["dev"]["descriptive"]["1"]["pd"] for m in MODES}
    reference = {"Mint-M0":100*(points["Mint"]-points["M0"]), "Mint-Msep":100*(points["Mint"]-points["Msep"])}
    pair_csv, pair_counts = _paired_csv(identities)
    _write(run_dir/"S4_paired_targets.csv",pair_csv)
    _write(run_dir/"S4_comparison.csv",_comparison_csv(runs))
    result = {"stage":"S4", "status":"COMPLETE", "execution_status":"COMPLETE", "step":step,
              "research_outcome":hypothesis, "reference_gain_pp":reference,"seeds":[46],
              "cross_seed_variability":"not assessed", "population":"NUDT-MIRSDT synthetic released sequences",
              "historical_test_scope":"17-frame valid-center subset, 1680 of 2000",
              "protocol_sha256":_sha(run_dir/"protocol.json"),"budget_decision_sha256":_sha(run_dir/"budget_decision.json"),
              "historical_lock_sha256":_sha(run_dir/"historical_lock.json"),"paired_gt_counts":pair_counts,
              "comparison_csv_sha256":_sha(run_dir/"S4_comparison.csv"),
              "paired_csv_sha256":_sha(run_dir/"S4_paired_targets.csv"),"runs":runs}
    migration = p.get("migration")
    if migration:
        parent = parent_run(run_dir,p)
        prior = _read(parent / "protocol.json")
        def chunks(protocol):
            q,v = runtime_chunks(protocol,"Mint")
            return f"q{q}/v{v}"
        result["migration"] = {**migration, "parent_chunks":chunks(prior), "child_chunks":chunks(p)}
    _save(run_dir/"S4_summary.json",result)
    _write(run_dir/"S4_research_report.md",_markdown(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--step",type=int,choices=(24000,36000))
    args = parser.parse_args()
    if args.step is None:
        print(decide_budget(args.run_dir)["selected_step"])
    else:
        print(emit_summary(args.run_dir,args.step)["execution_status"])


