"""Publish the fixed-seed S4 progress and paired evaluation milestones using committed local outputs."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time

if __package__:
    from .s4_lineage import diagnostic_origin, parent_run, runtime_chunks
else:
    from s4_lineage import diagnostic_origin, parent_run, runtime_chunks


MODES = ("Msep", "Mraw", "Mint", "M0")
SEED = 46
TERMS = ("loss", "foreground_bce", "background_bce_all", "background_bce_hard")
PAIRS = (("Mraw", "Msep"), ("Mint", "Mraw"), ("Mint", "M0"))
STEPS = (6000, 12000, 18000, 24000, 30000, 36000)


def read(path):
    return json.loads(path.read_bytes()) if path.is_file() else {}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"
    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def link(text, path):
    return f"[{text}](../{path})"


def last_update(path):
    """Read the last newline-committed record in the append-only update log."""
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        end = f.seek(0, os.SEEK_END)
        window = 8192
        while end:
            start = max(0, end - window)
            f.seek(start)
            data = f.read(end - start)
            done, _, _ = data.rpartition(b"\n")
            _, boundary, last = done.rpartition(b"\n")
            if boundary or not start:
                return json.loads(last) if last else {}
            window *= 2
    return {}


def mode_folder(run, mode):
    return run / mode / f"seed{SEED}"


def match(row, mode, step, protocol_sha):
    return (row.get("mode"), row.get("seed"), row.get("step"), row.get("protocol_sha256")) == (mode, SEED, step, protocol_sha)


def bound_diag(folder, mode, step, protocol_sha, protocol):
    source, protocol_sha = diagnostic_origin(folder.parents[1], protocol, mode, step)
    folder = mode_folder(source, mode)
    path = folder / f"diagnostics_{step}.json"
    j = read(path)
    if not j:
        return {}
    side = read(folder / "checkpoints" / f"step_{step:06d}.json")
    fit = j.get("fit", {})
    def core(row):
        return row["source"], row["center"], tuple(row["core_xyxy"])
    if (not match(j, mode, step, protocol_sha) or side.get("checkpoint_sha256") != j.get("checkpoint_sha256")
            or j.get("diagnostic_sha256") != protocol["diagnostics_sha256"]["train_diagnostics.json"]
            or fit.get("cores") != 144 or fit.get("finite") is not True or len(fit["by_core"]) != 144
            or {core(r) for r in fit["by_core"]} != {core(r) for r in protocol["fit_centers"]}):
        raise ValueError(f"训练诊断摘要或面板未通过核对 {path}")
    return fit


def bound_point(folder, mode, step, panel_ids, protocol_sha, execution_sha, population, protocol, history_sha=None):
    split = "test" if history_sha else "dev"
    out = folder / f"{split}_{step}"
    binding = read(out / "binding.json")
    if not binding:
        return None
    checkpoint = read(folder / "checkpoints" / f"step_{step:06d}.json")
    metrics = out / "metrics.json"
    expected_ids = set(panel_ids)
    ordered_sha = sha256("\n".join(panel_ids).encode()).hexdigest()
    if ((binding["query_chunk"], binding["velocity_chunk"]) != runtime_chunks(protocol,mode)
            or not match(binding, mode, step, protocol_sha) or binding.get("execution_binding_sha256") != execution_sha
            or binding.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256")
            or binding.get("population") != population or set(binding["frame_sha256"]) != expected_ids
            or binding.get("center_ids_sha256") != ordered_sha
            or (history_sha and binding.get("historical_lock_sha256") != history_sha)
            or digest(metrics) != binding.get("metrics_sha256")):
        raise ValueError(f"评价摘要未通过核对 {out}")
    doc = read(metrics)
    point = doc["transferred"]["1"] if history_sha else doc["descriptive"]["budgets"]["1"]
    if (doc.get("population") != population or set(point["per_frame"]) != expected_ids
            or doc["descriptive"]["frames"] != len(panel_ids)):
        raise ValueError(f"评价中心身份未通过核对 {out}")
    return point


def loss_text(value, initial):
    if value is None:
        return "—"
    if initial in (None, 0):
        return f"{value:.5f}"
    return f"{value:.5f} ({100*(value-initial)/initial:+.2f}%)"


def render(run):
    protocol = read(run / "protocol.json")
    if protocol.get("seeds") != [46] or set(protocol.get("modes", ())) != set(MODES):
        raise ValueError("此监测入口仅适用于 S4 固定种子 46 的四组正式配置")
    ps = digest(run / "protocol.json")
    es = digest(run / "execution_binding.json")
    driver = read(run / "driver/state.json")
    decision = read(run / "budget_decision.json")
    if decision and (decision.get("protocol_sha256") != ps or decision.get("selected_step") not in (24000, 36000)):
        raise ValueError("共同预算决定与本轮协议不符")
    end = decision.get("selected_step")
    lines = ["# S4 训练与评估进展", "",
             f"固定种子 46 · 更新于 {datetime.now().astimezone().isoformat(timespec='seconds')} · 运行 `{run.name}`", ""]
    migration = protocol.get("migration")
    if migration:
        parent = parent_run(run,protocol)
        previous = read(parent / "protocol.json")
        q0,v0 = runtime_chunks(previous,"Mint")
        q,v = runtime_chunks(protocol,"Mint")
        lines += [f"三个证据模型继承 [父运行](../../{parent.name}/protocol.json) 的前 {migration['common_step']:,} 步，"
                  f"从 q{q0}/v{v0} 切换至 q{q}/v{v}。M0 从配对初始状态训练。",""]
    final = read(run / "S4_summary.json") if end and (run / "historical_lock.json").is_file() else {}
    if final.get("execution_status") == "COMPLETE" and final.get("protocol_sha256") == ps and (run / "S4_research_report.md").is_file():
        lines += ["**正式报告已生成。** " + link("研究报告", "S4_research_report.md") + " · "
                  + link("四组完整比较", "S4_comparison.csv") + " · " + link("机器汇总", "S4_summary.json"), ""]
    if driver.get("status") == "FAILED":
        lines += ["**调度器记录了失败。** " + str(driver.get("error", "")) + " " + link("状态记录", "driver/state.json"), ""]
    phases = {"training": "训练中", "train_diagnostics": "计算固定训练诊断", "development_monitor": "评价 32 帧开发面板",
              "development": "评价 672 帧完整开发集", "training_target_complete": "已完成当前训练目标"}
    progress, loss_rows, points, current = [], [], {}, {}
    for mode in MODES:
        folder = mode_folder(run, mode)
        status = read(folder / "progress.json")
        row = last_update(folder / "updates.jsonl")
        inherited = migration["common_step"] if migration and mode in migration["modes"] else 0
        own_step = max(row.get("step", 0), status.get("step", 0))
        current[mode] = max(own_step, inherited)
        diags = {}
        for step in range(0, min(current[mode], end or 36000) + 1, 3000):
            fit = bound_diag(folder, mode, step, ps, protocol)
            if fit:
                diags[step] = fit
                base = diags.get(0, {})
                source = f"{step}（父运行）" if inherited and step <= inherited else step
                loss_rows.append([mode, source, *(loss_text(fit.get(key), base.get(key)) for key in TERMS)])
        phase = phases.get(status.get("phase"), "等待首条训练记录")
        if inherited and own_step <= inherited and not status:
            phase = "父运行前缀已锁定，等待继续训练"
        if status.get("phase") == "training_target_complete" and current[mode] > status.get("step", 0):
            phase = "已恢复训练"
        completed = end and match(read(folder / f"training_{end}.json"), mode, end, ps)
        if completed:
            phase = "共同终点已提交"
        last_diag = max(diags) if diags else None
        next_diag = last_diag + 3000 if last_diag is not None else 0
        waiting = "已齐" if end and last_diag == end else str(next_diag)
        progress.append([mode, current[mode], phase, str(last_diag) if last_diag is not None else "等待", waiting])
    lines += ["## 当前进度", "", *table(("模型","最近更新步","状态","最近已提交诊断","下一项诊断步"),progress), "",
              "## 固定 144 个训练裁块", "",
              "括号内是本模型相对初始化的变化。负值表示该损失下降。", "",
              *(table(("模型","步数","总损失","前景 BCE","全部背景 BCE","困难背景 BCE"),loss_rows)
                if loss_rows else ["等待四组各自提交首份 144 核诊断。"]), ""]
    for step in STEPS:
        if end and step > end:
            continue
        full = step % 12000 == 0
        ids = protocol["dev_ids" if full else "diagnostic_dev_ids"]
        if len(set(ids)) != len(ids) or len(ids) != (672 if full else 32):
            raise ValueError("开发面板中心数与冻结协议不符")
        points[step] = {m:bound_point(mode_folder(run,m),m,step,ids,ps,es,"development" if full else "development_monitor",protocol) for m in MODES}
    lines += ["## 开发集工作点", "", "6,000 步首次评价 32 帧监测面板，12,000 步首次评价 672 帧完整开发集。两种面板各自比较。格内为 1 FP/帧预算下的 TP / FP / Pd，Pd 以百分比表示。", ""]
    panel_rows, delta_rows = [], []
    for step, by_mode in points.items():
        full = step % 12000 == 0
        title = "672 帧完整开发集" if full else "32 帧固定监测"
        optional = "（条件延长）" if not end and step > 24000 else ""
        panel_rows.append([f"{step}{optional}", title, *[f"{p['tp']} / {p['fp']} / {100*p['pd']:.2f}%" if p else "等待" for p in by_mode.values()]])
        deltas = [f"{100*(by_mode[a]['pd']-by_mode[b]['pd']):+.2f}" if by_mode[a] and by_mode[b] else "等待" for a,b in PAIRS]
        delta_rows.append([step, title, *deltas])
    lines += [*table(("步数","面板",*MODES), panel_rows), "", "同一步数、同一面板的检出率差，单位为百分点。", "",
              *table(("步数","面板","Mraw − Msep","Mint − Mraw","Mint − M0"),delta_rows), "", "## 共同预算与历史测试", ""]
    if not end:
        lines += ["等待四个模型完成 24,000 步。预算判据使用固定训练面板在 18,000、21,000、24,000 步的诊断。"]
    else:
        lines += [f"共同终点为 **{end:,} 步**。 " + link("预算决定", "budget_decision.json")]
        # Access historical results only after all four selected development endpoints and the common lock are bound.
        lock_path = run / "historical_lock.json"
        lock = read(lock_path) if all(points[end].values()) else {}
        if (lock.get("state") == "HISTORICAL_LOCKED" and lock.get("selected_step") == end
                and lock.get("protocol_sha256") == ps and lock.get("execution_binding_sha256") == es
                and lock.get("budget_decision_sha256") == digest(run / "budget_decision.json") and set(lock["runs"]) == set(MODES)):
            historical_sha = digest(lock_path)
            for m in MODES:
                endpoint = mode_folder(run,m)
                bound = lock["runs"][m]["46"]
                train = read(endpoint / f"training_{end}.json")
                if (not match(train,m,end,ps) or bound["checkpoint_sha256"] != train["checkpoint_sha256"]
                        or bound["dev_metrics_sha256"] != digest(endpoint / f"dev_{end}/metrics.json")
                        or bound["thresholds"]["1"] != points[end][m]["threshold"]):
                    raise ValueError(f"历史锁与开发终点不符 {m}")
            history = []
            for m in MODES:
                p = bound_point(mode_folder(run,m),m,end,protocol["test_ids"],ps,es,"historical_test",protocol,historical_sha)
                history.append([m,"等待" if p is None else link("已提交", f"{m}/seed46/test_{end}/metrics.json"),
                                *(("—","—","—") if p is None else (p["tp"],p["fp"],f"{100*p['pd']:.2f}%"))])
            lines += ["", "1,680 帧历史测试沿用完整开发集阈值，实际虚警数可超过开发预算。 " + link("共同历史锁","historical_lock.json"),
                      "", *table(("模型","历史测试","TP","FP","Pd"),history)]
        else:
            lines += ["", "等待全部共同终点与开发阈值锁定。"]
    finished = (final.get("execution_status") == "COMPLETE" and final.get("protocol_sha256") == ps
                and (run / "S4_research_report.md").is_file())
    return "\n".join(lines) + "\n", finished


def refresh(run):
    content, finished = render(run)
    parent = run / "monitor"
    parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=parent, encoding="utf-8", delete=False) as stream:
        path = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(path, parent / "MILESTONES.md")
        finally:
            path.unlink(missing_ok=True)
    return finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Refresh once (default)")
    mode.add_argument("--watch", type=int, metavar="SECONDS", help="Refresh until the formal summary is complete")
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error("--watch must be a positive number of seconds")
    while True:
        if refresh(args.run_dir.resolve()) or not args.watch:
            break
        time.sleep(args.watch)
