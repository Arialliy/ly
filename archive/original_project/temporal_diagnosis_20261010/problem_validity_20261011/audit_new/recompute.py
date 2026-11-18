#!/usr/bin/env python3
"""Independent joins of cached detections. No component scoring or GPU use."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parents[1]
P = V.parent / 'stage1_goal'
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
ORIGINAL = {0, 36, 72, 108, 120}
SOURCES = {}


def digest(path):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    SOURCES[str(path)] = value
    return value


def read(directory, name):
    mp = directory / (name + '_COMPLETE.json')
    npz = directory / (name + '_matches.npz')
    digest(mp)
    digest(npz)
    info = json.loads(mp.read_text())
    assert info['status'] == 'COMPLETE'
    with np.load(npz, allow_pickle=False) as data:
        indices = np.flatnonzero(data['thresholds'] == 0.5)
        assert len(indices) == 1
        tidx = int(indices[0])
        matched = data['matched'][tidx]
        small = data['strict_small']
        box = data['box_max']
        assert len(info['objects']) == len(matched)
        result = {}
        for index, obj in enumerate(info['objects']):
            key = (obj['frame'], obj['annotation_id'])
            assert key not in result
            assert bool(obj['strict_small']) == bool(small[index])
            result[key] = {
                'object': obj, 'matched': bool(matched[index]),
                'box': float(box[index]),
            }
    return result


def count(records):
    result = collections.Counter()
    result['n'] = len(records)
    for row in records:
        f, a, b = row['f'], row['a'], row['b']
        result[f'F{int(f)}A{int(a)}'] += 1
        result['fused_matched'] += f
        result['fused_unmatched'] += not f
        result['any11_matched'] += a
        result['any5_matched'] += b
        result['fused_and_window_match'] += f and a
        result['fused_only_match'] += f and not a
        result['original_window_match_lost_by_fusion'] += not f and b
        result['extra_window_only_opportunity'] += not f and a and not b
        result['no_match_but_box_response'] += not f and not a and row['boxhigh']
        result['no_match_all_boxes_low'] += not f and not a and not row['boxhigh']
        result['any11_yes_fused_no'] += not f and a
        result['any11_no_fused_no'] += not f and not a
        result['final_detected_despite_response_cross'] += f and row['cross']
        result['final_detected_despite_window_match_disagreement'] += f and row['mixed']
    for key in ['F0A0', 'F0A1', 'F1A0', 'F1A1']:
        result.setdefault(key, 0)
    result['tracks'] = len({(r['seq'], r['track']) for r in records})
    result['fusion_loss_tracks'] = len({(r['seq'], r['track']) for r in records if not r['f'] and r['b']})
    result['extra_opportunity_tracks'] = len({(r['seq'], r['track']) for r in records if not r['f'] and r['a'] and not r['b']})
    assert sum(result[k] for k in ['F0A0', 'F0A1', 'F1A0', 'F1A1']) == len(records)
    assert result['F0A1'] == result['original_window_match_lost_by_fusion'] + result['extra_window_only_opportunity']
    assert result['F0A0'] == result['no_match_but_box_response'] + result['no_match_all_boxes_low']
    return dict(result)


def main():
    records = []
    for seq in SEQUENCES:
        fused = read(P / 'evaluation/official' / seq, 'official_fused')
        contexts = {s: read(V / 'window_scores' / seq, f'window_{s:03d}') for s in range(0, 121, 12)}
        for start, context in contexts.items():
            assert set(context) == {key for key in fused if start <= key[0] < start + 40}
        for key, target in fused.items():
            obj = target['object']
            windows = [(s, rows[key]) for s, rows in contexts.items() if key in rows]
            assert windows
            for _, row in windows:
                assert row['object'] == obj
                assert np.isfinite(row['box'])
            detections = [s for s, row in windows if row['matched']]
            values = [row['box'] for _, row in windows]
            records.append(dict(
                seq=seq, track=obj['track_id'], small=bool(obj['strict_small']),
                multi=len(windows) >= 2, f=target['matched'], a=bool(detections),
                b=bool(set(detections) & ORIGINAL), boxhigh=max(values) >= 0.5,
                cross=min(values) < 0.5 <= max(values),
                mixed=0 < len(detections) < len(windows),
            ))
    tables = []
    for seq in ['ALL'] + SEQUENCES:
        selected = [r for r in records if seq == 'ALL' or r['seq'] == seq]
        for scope in ['all_valid', 'small_geometry', 'all_valid_multi', 'small_geometry_multi']:
            subset = [r for r in selected if (not scope.startswith('small') or r['small']) and (not scope.endswith('_multi') or r['multi'])]
            tables.append(dict(sequence=seq, scope=scope, **count(subset)))
    out = V / 'audit_new'
    own = {'status': 'RECOMPUTED', 'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'tables': tables, 'sources': SOURCES, 'script_sha256': digest(__file__),
           'sample_status': 'Previously used calibration development materials, not independent confirmation.',
           'method': 'Independent frame/annotation-ID join from fused NPZ at 0.5 and 22 per-window NPZ; no scoring-code imports.'}
    (out / 'INDEPENDENT_COUNTS.json').write_text(json.dumps(own, indent=2) + '\n')
    summary_path = V / 'SUMMARY.json'
    if not summary_path.exists():
        print(json.dumps({'status': 'WAITING_ROOT_SUMMARY', 'tables': tables}))
        return
    summary = json.loads(summary_path.read_text())
    compared = 0
    mismatches = []
    for table in tables:
        expected = [r for r in summary['tables'] if r['group'] == 'additional_annotation_selected_development' and r['sequence'] == table['sequence'] and r['scope'] == table['scope']]
        assert len(expected) == 1
        expected = expected[0]
        for key, value in table.items():
            if key.startswith('F'):
                continue
            compared += 1
            if expected.get(key) != value:
                mismatches.append({'sequence': table['sequence'], 'scope': table['scope'], 'field': key, 'root': expected.get(key), 'independent': value})
    result = dict(status='PASS' if not mismatches else 'FAIL', compared_fields=compared,
                  tables_compared=len(tables), root_summary_sha256=digest(summary_path),
                  mismatches=mismatches, independent_tables=tables, sources=SOURCES,
                  sample_status=own['sample_status'])
    (out / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 两段补充开发材料的独立联表复算', '',
             f'结论：**{result["status"]}**。逐序列及合计、全部合法对象/几何小目标及多窗口子层，共核对 {len(tables)} 行、{compared} 个字段，差异 {len(mismatches)} 项。', '',
             'DJI_0026_2 和 DJI_0067_2 是按标注规则加入、先前用于校准的开发材料；不是新的独立确认集。', '',
             '从原融合 NPZ 的 0.5 阈值行和 22 个窗口 NPZ，按 `(frame, annotation_id)` 独立连接；逐项核对对象元数据、窗口覆盖、几何小目标标志。未导入主联表脚本，未重算连通组件，未访问确认集预测，未使用 GPU。', '',
             'F 表示原五窗最终融合匹配，A 表示任意 11 窗匹配；“原五窗有”属于已计算窗口中的融合后匹配损失，“仅额外窗”属于更多计算发现的机会。后者不是同预算可部署收益。', '',
             '| 序列 | 对象层 | N | F1 A1 | F1 A0 | F0 A1 | 其中原五窗有 | 其中仅额外窗 | F0 A0 | 其中框高响应 | 其中全低响应 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in tables:
        if r['scope'].endswith('_multi'):
            continue
        columns = [r['sequence'], r['scope']] + [r[k] for k in ['n', 'F1A1', 'F1A0', 'F0A1', 'original_window_match_lost_by_fusion', 'extra_window_only_opportunity', 'F0A0', 'no_match_but_box_response', 'no_match_all_boxes_low']]
        lines.append('| ' + ' | '.join(map(str, columns)) + ' |')
    lines.extend(['', '输入及输出哈希见 AUDIT.json；复算脚本为 recompute.py。以上均为诊断点匹配，不是论文 Pd/Fa。'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'compared_fields', 'tables_compared', 'mismatches']}))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
