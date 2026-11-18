#!/usr/bin/env python3
"""CPU-only reproducible checks of supplemental report claims from object tables."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
V = OUT.parent
SEQUENCES = {'DJI_0026_2', 'DJI_0067_2'}
ORIGINAL = {0, 36, 72, 108, 120}
MODES = ['official5_probmean', 'symmetric5_probmax']
SOURCES = {}


def load_csv(relative):
    path = V / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


def key(row):
    return row['sequence'], int(row['frame']), int(row['annotation_id'])


def bit(row, field):
    value = int(row[field])
    assert value in [0, 1], (field, value)
    return bool(value)


def write_csv(name, records):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    original_rows = [r for r in load_csv('OBJECTS.csv') if r['sequence'] in SEQUENCES]
    original = {key(r): r for r in original_rows}
    assert len(original) == len(original_rows)
    windows = collections.defaultdict(list)
    for row in load_csv('WINDOWS.csv'):
        if row['sequence'] in SEQUENCES:
            windows[key(row)].append(row)
    assert set(windows) == set(original)
    controls = {}
    for row in load_csv('control_summary/OBJECTS.csv'):
        assert row['sequence'] in SEQUENCES and row['mode'] in MODES
        control_key = (row['mode'], key(row))
        assert control_key not in controls
        controls[control_key] = row
    for mode in MODES:
        assert {k[1] for k in controls if k[0] == mode} == set(original)
    declared_tracks = {}
    for row in load_csv('control_summary/TRACKS.csv'):
        tkey = row['mode'], row['sequence'], int(row['track_id'])
        assert tkey not in declared_tracks
        declared_tracks[tkey] = {k: int(row[k]) for k in ['n', 'gained', 'lost', 'net']}

    stable = []
    extra = []
    tracks = collections.defaultdict(lambda: {'n': 0, 'gained': 0, 'lost': 0, 'net': 0})
    for objkey, obj in original.items():
        wr = windows[objkey]
        starts = [int(w['window_start']) for w in wr]
        expected_starts = [s for s in range(0, 121, 12) if s <= objkey[1] < s + 40]
        assert sorted(starts) == expected_starts
        for w in wr:
            assert int(w['slot']) == objkey[1] - int(w['window_start'])
            assert bit(w, 'original5') == (int(w['window_start']) in ORIGINAL)
            assert int(w['track_id']) == int(obj['track_id'])
            assert w['source_frame'] == obj['source_frame']
            assert bit(w, 'fused_matched') == bit(obj, 'fused_matched')
        successful = [w for w in wr if bit(w, 'matched')]
        any_match = bool(successful)
        all_match = len(successful) == len(wr)
        official_match = any(bit(w, 'original5') for w in successful)
        assert bit(obj, 'any11_matched') == any_match
        assert bit(obj, 'any5_matched') == official_match
        assert bit(obj, 'match_disagreement') == (any_match and not all_match)
        assert bit(obj, 'multi_context') == (len(wr) >= 2)
        assert int(obj['window_count']) == len(wr)
        small = bit(obj, 'small_geometry')
        fused = bit(obj, 'fused_matched')

        cr = {}
        for mode in MODES:
            row = controls[(mode, objkey)]
            assert row['source_frame'] == obj['source_frame'] and row['track_id'] == obj['track_id']
            assert bit(row, 'small_geometry') == small
            assert bit(row, 'official_matched') == fused
            detected = bit(row, 'control_matched')
            gained = int(detected and not fused)
            lost = int(fused and not detected)
            assert int(row['gain']) == gained and int(row['loss']) == lost
            cr[mode] = {'matched': int(detected), 'lost': lost}
            if small:
                counter = tracks[(mode, objkey[0], int(obj['track_id']))]
                counter['n'] += 1
                counter['gained'] += gained
                counter['lost'] += lost
                counter['net'] += gained - lost

        if small and len(wr) >= 2 and all_match and fused:
            record = dict(sequence=objkey[0], frame=objkey[1], annotation_id=objkey[2],
                          track_id=int(obj['track_id']), eligible_windows=len(wr))
            for mode in MODES:
                record[mode + '_matched'] = cr[mode]['matched']
                record[mode + '_lost'] = cr[mode]['lost']
            stable.append(record)
        if small and not fused and any_match and not official_match:
            assert obj['system_case'] == 'extra_window_only_opportunity'
            original_inner = any(bit(w, 'original5') and 8 <= int(w['slot']) <= 31 for w in wr)
            successful_inner = any(8 <= int(w['slot']) <= 31 for w in successful)
            extra.append(dict(sequence=objkey[0], frame=objkey[1], annotation_id=objkey[2],
                              track_id=int(obj['track_id']), original_has_internal_slot=int(original_inner),
                              successful_has_internal_slot=int(successful_inner),
                              original_internal_starts='|'.join(str(w['window_start']) for w in wr if bit(w, 'original5') and 8 <= int(w['slot']) <= 31),
                              successful_internal_starts='|'.join(str(w['window_start']) for w in successful if 8 <= int(w['slot']) <= 31)))

    assert set(tracks) == set(declared_tracks)
    track_mismatches = [dict(mode=k[0], sequence=k[1], track_id=k[2], declared=declared_tracks[k], recomputed=v)
                        for k, v in tracks.items() if v != declared_tracks[k]]
    per_track = [dict(mode=k[0], sequence=k[1], track_id=k[2], **v) for k, v in sorted(tracks.items())]
    net_counts = {}
    for mode in MODES:
        selected = [v for k, v in tracks.items() if k[0] == mode]
        net_counts[mode] = dict(tracks=len(selected), net_positive=sum(r['net'] > 0 for r in selected),
                                net_negative=sum(r['net'] < 0 for r in selected), net_zero=sum(r['net'] == 0 for r in selected),
                                gained=sum(r['gained'] for r in selected), lost=sum(r['lost'] for r in selected))
    stable_by_seq = {seq: sum(r['sequence'] == seq for r in stable) for seq in sorted(SEQUENCES)}
    stable_losses = {mode: sum(r[mode + '_lost'] for r in stable) for mode in MODES}
    slot_counts = dict(extra_opportunities=len(extra),
                       original_has_internal_slot=sum(r['original_has_internal_slot'] for r in extra),
                       successful_has_internal_slot=sum(r['successful_has_internal_slot'] for r in extra),
                       both=sum(r['original_has_internal_slot'] and r['successful_has_internal_slot'] for r in extra),
                       neither=sum(not r['original_has_internal_slot'] and not r['successful_has_internal_slot'] for r in extra))
    observed = {'stable_n': len(stable), 'stable_control_losses': stable_losses,
                'extra_opportunity_slots': slot_counts, 'track_net_counts': net_counts}
    checks = [dict(claim='stable_n', expected=422, actual=len(stable)),
              dict(claim='mean_stable_loss', expected=0, actual=stable_losses['official5_probmean']),
              dict(claim='sym_stable_loss', expected=0, actual=stable_losses['symmetric5_probmax']),
              dict(claim='extra_n', expected=104, actual=len(extra)),
              dict(claim='extra_original_internal', expected=59, actual=slot_counts['original_has_internal_slot']),
              dict(claim='extra_success_internal', expected=77, actual=slot_counts['successful_has_internal_slot'])]
    for mode, expected in [('symmetric5_probmax', (14, 13, 103)), ('official5_probmean', (1, 19, 110))]:
        for field, value in zip(['net_positive', 'net_negative', 'net_zero'], expected):
            checks.append(dict(claim=mode + '_' + field, expected=value, actual=net_counts[mode][field]))
    for check in checks:
        check['pass'] = check['expected'] == check['actual']
    result = dict(status='PASS' if all(c['pass'] for c in checks) and not track_mismatches else 'FAIL',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  threshold=0.5, sample_status='Previously used calibration development materials; not independent confirmation.',
                  observations=observed, stable_by_sequence=stable_by_seq,
                  checks=checks, track_mismatches=track_mismatches,
                  source_sha256=SOURCES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_csv('STABLE_OBJECTS.csv', stable)
    write_csv('EXTRA_OPPORTUNITIES.csv', extra)
    write_csv('TRACK_NET.csv', per_track)
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 补充诊断数字的独立复算', '',
             f'核对结果：**{result["status"]}**。12 项报告数字核对，{sum(not c["pass"] for c in checks)} 项不一致；逐轨迹计数与 control_summary/TRACKS.csv 的差异为 {len(track_mismatches)} 项。', '',
             '数据为 DJI_0026_2 与 DJI_0067_2，均是此前用于校准的开发材料，不是新的独立确认集。输入来自根 OBJECTS.csv、WINDOWS.csv，以及 control_summary/OBJECTS.csv、TRACKS.csv。本程序独立重建集合与逐轨迹计数；没有重新计算检测组件，也未访问 GPU。', '',
             '所有控制数值对应固定阈值 0.5。读取逐对象匹配，重新计算新增与丢失，并与表内 gain/loss 校对。', '',
             '## 稳定检出集合', '',
             '集合要求：几何小目标、当前帧至少被两个诊断窗口覆盖、全部可用诊断窗口均匹配、原五窗融合也匹配。这里“全部”指覆盖当前帧的可用窗口，不要求每帧都被 11 个窗口覆盖。逐窗记录独立验证了 any11_matched、match_disagreement 与 multi_context 字段。', '',
             f'- 集合大小：{len(stable)}；逐序列为 {stable_by_seq}。',
             f'- 原五窗概率均值控制新增丢失：{stable_losses["official5_probmean"]}。',
             f'- 对称五窗概率 max 控制新增丢失：{stable_losses["symmetric5_probmax"]}。', '',
             '逐对象证据保存在 STABLE_OBJECTS.csv。此集合全保留不能推出所有原本检出的对象都被保留。', '',
             '## 额外窗口机会与内部槽位', '',
             '集合要求：几何小目标、原融合未匹配、任一诊断窗匹配、原五窗中没有任何匹配。内部槽位定义为 8–31（含端点），通过逐窗 frame−window_start 重新计算。', '',
             f'- 额外窗口机会：{slot_counts["extra_opportunities"]}。',
             f'- 原五窗中已有内部槽位：{slot_counts["original_has_internal_slot"]}。',
             f'- 成功窗口中存在内部槽位：{slot_counts["successful_has_internal_slot"]}。',
             f'- 两条件同时满足：{slot_counts["both"]}；均不满足：{slot_counts["neither"]}。', '',
             '这两个条件可以重叠，不能相加为互斥分类。自然窗口内容与槽位共同变化，以上计数不是位置的因果效应；更多窗口提供的机会也不是同计算预算可实现的收益。逐对象证据在 EXTRA_OPPORTUNITIES.csv。', '',
             '## 轨迹净变化', '',
             '以 (sequence, track_id) 为轨迹键，只纳入几何小目标帧。逐轨迹净值 = 新增匹配帧数 − 丢失匹配帧数；净零可能包含新增与丢失抵消。', '',
             '| 控制 | 轨迹数 | 净正 | 净负 | 净零 | 新增帧合计 | 丢失帧合计 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        row = net_counts[mode]
        lines.append('| ' + ' | '.join([mode] + [str(row[k]) for k in ['tracks', 'net_positive', 'net_negative', 'net_zero', 'gained', 'lost']]) + ' |')
    lines += ['', '逐轨迹数据在 TRACK_NET.csv；输入文件 SHA256 与全部核对结果在 RESULTS.json。', '',
              '复算命令（仅 Python 标准库、CPU）：', '', '```bash',
              f'python3 {Path(__file__).resolve()}', '```', '',
              '以上是诊断点匹配，不是论文 Pd/Fa。不以相邻目标帧作为独立样本做显著性检验。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'observations': observed, 'track_mismatches': track_mismatches}, ensure_ascii=False))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
