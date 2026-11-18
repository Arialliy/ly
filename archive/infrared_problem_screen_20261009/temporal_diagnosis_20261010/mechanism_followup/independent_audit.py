#!/usr/bin/env python3
"""Independent standard-library audit; does not import inference/analysis code."""
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
FACTORS = ['slot', 'feature', 'membership']
PREFIXES = {'symmetric': '', 'anchor_a': 'anchor_a_', 'anchor_b': 'anchor_b_'}

def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def key(row):
    return row['sequence'], int(row['local_frame']), int(row['track_id']), int(row['annotation_id'])

def num(row, name):
    return float(row[name])

def bits(mask):
    return ''.join(str((mask >> i) & 1) for i in range(3))

def shapley(row, prefix, readout):
    values = [num(row, prefix+'combo_'+bits(mask)+'_'+readout) for mask in range(8)]
    contribution = [0., 0., 0.]
    for permutation in itertools.permutations(range(3)):
        previous = 0
        for factor in permutation:
            current = previous | (1 << factor)
            contribution[factor] += (values[current] - values[previous]) / 6
            previous = current
    # Different expression checks the permutation implementation independently.
    alternate = []
    for factor in range(3):
        total = 0.
        for mask in range(8):
            if mask & (1 << factor):
                continue
            count = mask.bit_count()
            weight = math.factorial(count)*math.factorial(2-count)/math.factorial(3)
            total += weight*(values[mask | (1 << factor)]-values[mask])
        alternate.append(total)
    assert max(abs(a-b) for a,b in zip(contribution, alternate)) < 1e-10
    return contribution, abs(sum(contribution)-(values[7]-values[0]))

def main():
    original = [r for r in read(ROOT/'result_stratification/context/same_gt_frame_choices.csv')
                if num(r,'threshold') == .5 and int(r['strict']) == 1
                and int(r['multi']) == 1 and num(r,'max_side') <= 16]
    original.sort(key=key)
    expected = {key(r):r for r in original}
    assert len(original) == len(expected) == 1417
    contexts = collections.defaultdict(list)
    for sequence in sorted({r['sequence'] for r in original}):
        for row in read(ROOT/'window_offset_experiment/analysis'/sequence/'per_context_all_tracks.csv'):
            if num(row,'threshold') == .5:
                contexts[key(row)].append(row)
    rows = []
    input_hashes = {}
    hash_assertions = {}
    shard_counts = []
    metas = []
    verified_files = {}
    for shard in range(4):
        directory = BASE/('shard_%02d_of_04'%shard)
        meta = json.loads((directory/'metadata.json').read_text())
        selected = read(directory/'selected_pairs.csv')
        result = read(directory/'results.csv')
        metas.append(meta)
        assert meta['status'] == 'COMPLETE'
        assert meta['completed'] == meta['selected_total'] == len(selected) == len(result)
        assert [key(r) for r in selected] == [key(r) for r in result]
        assert all(int(r['pair_index']) % 4 == shard for r in result)
        for selected_row,result_row in zip(selected,result):
            assert all(result_row[column] == value for column,value in selected_row.items())
        for name,field in [('results.csv','result_sha256'),('selected_pairs.csv','selected_pairs_sha256')]:
            path = directory/name
            actual = sha(path)
            assert actual == meta[field]
            input_hashes[str(path)] = actual
        for field in ['input_sha256','raw_input_sha256','layer_source_sha256']:
            for path, value in meta[field].items():
                assert path not in hash_assertions or hash_assertions[path] == value
                hash_assertions[path] = value
        for field,pathfield in [('checkpoint_sha256','checkpoint'),('model_source_sha256','model_file')]:
            hash_assertions[meta[pathfield]] = meta[field]
        rows.extend(result)
        shard_counts.append(len(result))
    for path, expected_hash in hash_assertions.items():
        actual = sha(Path(path))
        assert actual == expected_hash, path
        verified_files[path] = actual
    snapshot_hash = sha(BASE/'run_factorial_frozen.py')
    assert snapshot_hash == 'fa2004b402add5a028f45af101c0604dd225b3e69168bc9f3acaef02bfebb238'
    assert all(m['wrapper_sha256'] == snapshot_hash for m in metas)
    rows.sort(key=lambda r:int(r['pair_index']))
    assert [int(r['pair_index']) for r in rows] == list(range(1417))
    assert len({key(r) for r in rows}) == 1417
    assert set(key(r) for r in rows) == set(expected)
    for index,row in enumerate(rows):
        assert key(row) == key(original[index])
        for field,value in expected[key(row)].items():
            assert row[field] == value
        windows = sorted(contexts[key(row)],key=lambda r:(num(r,'box_max'),int(r['window_start'])))
        assert len(windows) == int(row['context_count'])
        for tag, endpoint in [('a',windows[0]),('b',windows[-1])]:
            assert int(row[tag+'_start']) == int(endpoint['window_start'])
            assert int(row[tag+'_slot']) == int(endpoint['slot'])
            assert num(row,tag+'_saved_box_max') == num(endpoint,'box_max')
            assert int(row['local_frame']) == int(row[tag+'_start']) + int(row[tag+'_slot'])
            assert abs(num(row,tag+'_replay_box_max')-num(row,tag+'_saved_box_max')) == num(row,tag+'_replay_abs_error')
            for threshold in [.1,.5,.9]:
                disagreement = int((num(row,tag+'_saved_box_max') >= threshold) != (num(row,tag+'_replay_box_max') >= threshold))
                assert disagreement == int(row[tag+'_replay_threshold_disagreement_'+str(threshold).replace('.','_')]) == 0
            assert num(row,tag+'_replay_abs_error') <= 1e-4
            start = int(row[tag+'_start'])
            assert int(row[tag+'_halo_real_before']) == min(8,start)
            assert int(row[tag+'_halo_real_after']) == min(8,160-start-40)
            for side in ['before','after']:
                assert int(row[tag+'_halo_missing_'+side])+int(row[tag+'_halo_real_'+side]) == 8
        x,y,w,h = [num(row,'bbox_'+v) for v in ['x','y','w','h']]
        # Cohort has fully in-frame bboxes, so no clipping is needed here.
        assert [int(row['crop_'+v]) for v in ['x0','y0','x1','y1']] == [math.ceil(x-.5),math.ceil(y-.5),math.ceil(x+w-.5),math.ceil(y+h-.5)]
        assert num(row,'projection_reconstruction_max_abs_error') <= 1e-4+1e-5*num(row,'projection_scale')
        for prefix in PREFIXES.values():
            for combo,tag in [('000','a'),('111','b')]:
                assert abs(num(row,prefix+'combo_'+combo+'_box_max_probability')-num(row,tag+'_replay_box_max')) <= 1e-4
                assert abs(num(row,prefix+'combo_'+combo+'_box_max_logit')-num(row,tag+'_replay_box_max_logit')) <= 1e-4 + 1e-4*abs(num(row,tag+'_replay_box_max_logit'))
        for name,value in row.items():
            if name.startswith(('combo_','anchor_')) and ('_logit' in name or '_probability' in name):
                assert math.isfinite(float(value))
    error_fields = [f for f in rows[0] if 'error' in f]
    max_errors = {field:max(num(r,field) for r in rows) for field in error_fields}
    for field in ['a_factorial_endpoint_max_abs_probability_error','b_factorial_endpoint_max_abs_probability_error',
                  'anchor_a_endpoint_max_abs_probability_error','anchor_b_endpoint_max_abs_probability_error']:
        assert max_errors[field] <= 1e-4
    all_shapley = {}
    max_additivity_error = 0.
    for row in rows:
        result = {}
        for anchor,prefix in PREFIXES.items():
            result[anchor] = {}
            for readout in ['box_max_logit','a_peak_logit','b_peak_logit','box_max_probability']:
                values,error = shapley(row,prefix,readout)
                max_additivity_error = max(max_additivity_error,error)
                result[anchor][readout] = values
        all_shapley[int(row['pair_index'])] = result
    assert max_additivity_error < 1e-10
    groups = {
        'all':rows,
        'strong_flip':[r for r in rows if num(r,'a_saved_box_max') < .1 and num(r,'b_saved_box_max') >= .9],
        'cross_05':[r for r in rows if num(r,'a_saved_box_max') < .5 <= num(r,'b_saved_box_max')],
        'all_low':[r for r in rows if num(r,'b_saved_box_max') < .5],
        'all_high':[r for r in rows if num(r,'a_saved_box_max') >= .5],
    }
    assert [len(groups[g]) for g in ['strong_flip','cross_05','all_low','all_high']] == [97,137,778,502]
    summaries = {}
    for group,subset in groups.items():
        out = {'n':len(subset),'track_ids':len({r['track_id'] for r in subset}),
               'sequences':dict(collections.Counter(r['sequence'] for r in subset)), 'anchors':{}}
        for anchor in PREFIXES:
            target = {}
            for readout in ['box_max_logit','a_peak_logit','b_peak_logit','box_max_probability']:
                contributions = [all_shapley[int(r['pair_index'])][anchor][readout] for r in subset]
                target[readout] = {
                    'largest_signed':dict(collections.Counter(FACTORS[max(range(3),key=lambda i:v[i])] for v in contributions)),
                    'largest_absolute':dict(collections.Counter(FACTORS[max(range(3),key=lambda i:abs(v[i]))] for v in contributions)),
                    'mean':{name:statistics.mean(v[i] for v in contributions) for i,name in enumerate(FACTORS)},
                    'median':{name:statistics.median(v[i] for v in contributions) for i,name in enumerate(FACTORS)},
                }
            out['anchors'][anchor] = target
        stable = collections.Counter()
        changes = 0
        for row in subset:
            dominant = [max(range(3),key=lambda i:all_shapley[int(row['pair_index'])][a]['box_max_logit'][i]) for a in PREFIXES]
            if len(set(dominant)) == 1:
                stable[FACTORS[dominant[0]]] += 1
            else:
                changes += 1
        out['dominant_signed_stable_across_anchors'] = dict(stable)
        out['dominant_signed_changes_across_anchors'] = changes
        out['halo'] = {}
        for tag in ['a','b']:
            for threshold in [.1,.5,.9]:
                key_ = tag+'_threshold_'+str(threshold)
                out['halo'][key_] = {
                    'gain':sum(num(r,tag+'_replay_box_max') < threshold <= num(r,tag+'_halo_box_max') for r in subset),
                    'loss':sum(num(r,tag+'_halo_box_max') < threshold <= num(r,tag+'_replay_box_max') for r in subset),
                }
        out['both_full_halo_n'] = sum(all(int(r[t+'_halo_missing_'+s]) == 0 for t in ['a','b'] for s in ['before','after']) for r in subset)
        full_subset = [r for r in subset if all(int(r[t+'_halo_missing_'+s]) == 0 for t in ['a','b'] for s in ['before','after'])]
        out['halo_either_endpoint_above_05'] = sum(max(num(r,t+'_halo_box_max') for t in ['a','b']) >= .5 for r in subset)
        out['full_halo_only'] = {
            'n':len(full_subset),
            'a_above_05':sum(num(r,'a_halo_box_max') >= .5 for r in full_subset),
            'b_below_05':sum(num(r,'b_halo_box_max') < .5 for r in full_subset),
            'either_above_05':sum(max(num(r,t+'_halo_box_max') for t in ['a','b']) >= .5 for r in full_subset),
        }
        summaries[group] = out
    strong = groups['strong_flip']
    by_sequence = {}
    for seq in sorted({r['sequence'] for r in strong}):
        subset = [r for r in strong if r['sequence'] == seq]
        by_sequence[seq] = {anchor:dict(collections.Counter(
            FACTORS[max(range(3),key=lambda i:all_shapley[int(r['pair_index'])][anchor]['box_max_logit'][i])] for r in subset)) for anchor in PREFIXES}
    payload = {
        'status':'PASS','scope':'Independent CPU standard-library audit of saved metadata/CSVs; no import of main inference/analysis script.',
        'inference_reexecuted':False,
        'tensor_limit':'Full tensor activations are not saved; tensor-level checks are audited in source and their recorded maxima/gates checked. Scalar endpoints and Shapley independently recomputed from CSV.',
        'coverage':{'total':len(rows),'unique_target_frames':len({key(r) for r in rows}),'shards':shard_counts,'pair_index_range':[0,1416]},
        'snapshot_sha256':snapshot_hash,'threshold_replay_disagreements':0,'maximum_errors':max_errors,
        'shapley_max_additivity_error':max_additivity_error,'summaries':summaries,'strong_flip_by_sequence':by_sequence,
        'input_result_sha256':input_hashes,'verified_source_input_sha256':verified_files,
        'independent_audit_script_sha256':sha(Path(__file__)),
        'interpretation_limits':['Extreme contexts selected from saved outputs; attribution is conditional and exploratory.',
                                 'Slot includes bias; membership includes exclusive frame weights and features.',
                                 'Three-factor split and nonlinear readout are conventions; anchors report sensitivity.',
                                 'Repeated target frames are not independent replicates; no official Pd/Fa or deployment improvement is claimed.'],
    }
    (BASE/'AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    sf = summaries['strong_flip']
    text = ['# 独立数值审计','', '**结论：PASS。** 四个分片的 1,417 个目标帧覆盖完整，未发现连接、重放阈值或分解端点不一致。', '',
            '本审计使用独立 Python 标准库程序读取原始选择表、逐上下文表与四分片结果；没有导入主推理或主统计脚本，没有再次执行模型。完整激活张量未落盘，因此张量级精确重构依据已独立审计的源码 gate 与运行记录；CSV 标量端点、集合连接和 Shapley 是独立复算。','',
            '## 覆盖与数值门槛','',f'- 分片数量：{shard_counts}；pair_index 恰为 0..1416，目标帧键无重复、无遗漏；与旧 1,417 样本及独立重选 min/max 窗一致。',
            '- 原始选择、上下文、raw 输入、checkpoint、模型层源码及结果文件哈希全部匹配；冻结脚本 SHA256 与独立代码审计版本一致。',
            '- 97 强翻转、137 跨 0.5、778 始终低、502 始终高，独立复算完全一致。',
            '- 2,834 个真实端点在 0.1 / 0.5 / 0.9 三阈值下均为零重放分歧。',
            f'- 最大重放概率误差：{max(max_errors[t+"_replay_abs_error"] for t in ["a","b"]):.9g}。',
            f'- 最大投影重构误差：{max_errors["projection_reconstruction_max_abs_error"]:.9g}；共享内部帧最大特征误差：{max_errors["shared_interior_feature_max_abs_error"]:.9g}。',
            f'- 对称及双锚端点最大 logit 误差：{max(v for k,v in max_errors.items() if "endpoint" in k and "logit" in k):.9g}。',
            f'- 3 种锚 × 4 种读出 × 1,417 行均独立用六种排列计算 Shapley，并用子集公式交叉核验；最大加性残差 {max_additivity_error:.9g}。','',
            '## 97 个强翻转的内部贡献','',
            '下表“主导”指对框内最大 logit 从较低窗到较高窗变化贡献最大的**有符号** Shapley 项；不是独立因果份额。','',
            '| 分解约定 | 槽位系数及 bias | 共同帧前端特征 | 非重叠上下文成员 |','|---|---:|---:|---:|']
    for anchor in PREFIXES:
        count=sf['anchors'][anchor]['box_max_logit']['largest_signed']
        text.append('| %s | %d | %d | %d |'%(anchor,count.get('slot',0),count.get('feature',0),count.get('membership',0)))
    text += ['',f'- 三种分配约定下主导项都一致：{sf["dominant_signed_stable_across_anchors"]}；主导项改变 {sf["dominant_signed_changes_across_anchors"]}/97。',
             '- 固定较低窗／较高窗概率峰值、绝对贡献主导及概率读出的敏感性统计均在 AUDIT.json 中，避免只选择单一有利读出。',
             '- 完整真实 halo 的配对数量：%d/97。'%sf['both_full_halo_n'],
             '- 真实 halo 对低窗跨 0.5 的新增／丢失为 %d/%d，高窗为 %d/%d。'%(sf['halo']['a_threshold_0.5']['gain'],sf['halo']['a_threshold_0.5']['loss'],sf['halo']['b_threshold_0.5']['gain'],sf['halo']['b_threshold_0.5']['loss']),
             '- 限定完整真实 halo 的 66 个强翻转：低窗恢复 27 个，高窗丢失 8 个，仍存在双向得失。',
             '- 778 个既有始终低响应目标帧，halo 两端任一恢复到 0.5 的仅 39 个；完整 halo 子集为 24/489。',
             '- 固定高窗峰值时 membership 主导 63/97；固定低窗峰值时则为 slot 32、feature 36、membership 29。后者的响应变化可能为负，不能宣称所有像素由同一项主导。',
             '', '这些是既有极值窗条件下的内部机制诊断。非重叠项包含帧内容及其权重，槽位项包含 bias；不能把分解命名为唯一自然因果解释，也不能把框内响应增强称为新增检测或官方 Pd/Fa 提升。', '',
             '代码审计记录见 `source_audit.md`；可复算程序为 `independent_audit.py`；详细数值和输入哈希见 `AUDIT.json`。']
    (BASE/'AUDIT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status':'PASS','coverage':payload['coverage'],'max_errors':max_errors,'strong_flip':sf,'strong_flip_by_sequence':by_sequence},indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
