"""M1-v1.3 same-weight source-bank traces using the supplied bank core.

ROIs are used only after model operations. All three modes share the same F/H
and checkpoint. No training, model mutation, spatial M1 tiling or hidden state.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F


MODES = ("History", "HistoryCurrent", "HistoryOff")
STAGES = ("previous_input", "after_pda", "after_tsfm", "residual", "injected_H",
          "head_conv1", "head_bn", "head_relu", "logit")


def _validate_head(base):
    assert len(base.conv_out1) == 3
    conv, bn, relu = base.conv_out1
    assert isinstance(conv, torch.nn.Conv3d) and isinstance(bn, torch.nn.BatchNorm3d)
    assert isinstance(relu, torch.nn.ReLU) and isinstance(base.conv_out2, torch.nn.Conv3d)
    for layer in (conv, base.conv_out2):
        assert layer.kernel_size == layer.stride == layer.dilation == (1, 1, 1)
        assert layer.padding == (0, 0, 0) and layer.groups == 1
    assert conv.in_channels == 32 and conv.out_channels == 8
    assert base.conv_out2.in_channels == 8 and base.conv_out2.out_channels == 1
    assert not base.training and not bn.training


def head_input_jacobian(base, bn_preactivation):
    """Exact local dz/dH_new for the frozen pointwise head, shape [...,32].

    bn_preactivation has last dimension8, before ReLU. The derivative at exactly
    zero uses PyTorch ReLU's convention0. This differentiates logits, not
    probabilities or loss, and does not request parameter gradients.
    """
    _validate_head(base)
    if bn_preactivation.shape[-1] != 8:
        raise ValueError("bn_preactivation must end in eight head channels")
    conv, bn, _ = base.conv_out1
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    coefficient = base.conv_out2.weight.reshape(8) * scale
    gates = (bn_preactivation > 0).to(coefficient.dtype)
    return (gates * coefficient) @ conv.weight.reshape(8, 32)


def _head(base, value):
    """Preserve the original operators and save BN before its inplace ReLU."""
    conv = base.conv_out1[0](value.unsqueeze(2).contiguous())
    bn = base.conv_out1[1](conv)
    relu = base.conv_out1[2](bn.clone())
    logit = base.conv_out2(relu)
    return conv.squeeze(2), bn.squeeze(2), relu.squeeze(2), logit.squeeze(2)


def _regions_at(regions, t, batch, height, width):
    supplied = regions.get(t, []) if isinstance(regions, Mapping) else regions[t]
    valid, skipped = [], 0
    for original in supplied:
        item = dict(original)
        b = int(item.get("batch_index", 0))
        if not 0 <= b < batch:
            raise ValueError("ROI batch_index is outside the current batch")
        box = [item[k] for k in ("x1", "y1", "x2", "y2")]
        if any(not isinstance(v, (int, np.integer)) for v in box):
            raise ValueError("ROIs use integer, half-open pixel coordinates")
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
        if x1 >= x2 or y1 >= y2:
            skipped += 1
            continue
        ax = int(item.get("anchor_x", (x1+x2-1)//2))
        ay = int(item.get("anchor_y", (y1+y2-1)//2))
        if not x1 <= ax < x2 or not y1 <= ay < y2:
            raise ValueError("Fixed ROI anchor must be inside the clipped ROI")
        item.update(batch_index=b, x1=x1, y1=y1, x2=x2, y2=y2,
                    anchor_x=ax, anchor_y=ay, roi_pixels=(x2-x1)*(y2-y1),
                    box_clipped=box != [x1,y1,x2,y2])
        valid.append(item)
    return valid, skipped


def _moments(history, current):
    delta = history-current
    return torch.stack((delta.mean(1), delta.square().mean(1), history.square().mean(1),
                        current.square().mean(1), delta.abs().amax(1)), 0)


def _statistics(values, maximum):
    mean_delta, square_delta, square_history, square_current = map(float, values)
    delta, history, current = [math.sqrt(max(0., v)) for v in (square_delta, square_history, square_current)]
    amplitude = math.sqrt(max(0., (square_history+square_current)/2))
    return {"delta_mean": mean_delta, "delta_rms": delta, "delta_absmax": float(maximum),
            "history_rms": history, "current_rms": current,
            "relative_delta_rms": delta / amplitude if amplitude > 1e-12 else None}


def _stage_statistics(maps, rois):
    """One CPU transfer/frame, vectorized ROI sums, no per-ROI CUDA sync.

    maps: [stage, moment, batch, height, width], moments as _moments above.
    Accumulate squared/signed moment maps in float64 to avoid subtractive errors
    when small ROIs are far from the integral-image origin.
    """
    global_means = maps[:, :4].mean(axis=(2,3,4), dtype=np.float64)
    global_max = maps[:, 4].max(axis=(1,2,3))
    global_values = {stage: _statistics(global_means[i], global_max[i]) for i,stage in enumerate(STAGES)}
    if not rois:
        return global_values, []
    moment = maps[:, :4]
    integral = np.pad(moment.cumsum(axis=-2, dtype=np.float64).cumsum(axis=-1, dtype=np.float64),
                      ((0,0),(0,0),(0,0),(1,0),(1,0)))
    b = np.array([r["batch_index"] for r in rois])
    x1, y1, x2, y2 = (np.array([r[k] for r in rois]) for k in ("x1","y1","x2","y2"))
    area = (x2-x1)*(y2-y1)
    means = (integral[:,:,b,y2,x2]-integral[:,:,b,y1,x2]
             -integral[:,:,b,y2,x1]+integral[:,:,b,y1,x1])/area
    roi_values = []
    for ri, roi in enumerate(rois):
        maxima = maps[:,4,roi["batch_index"],roi["y1"]:roi["y2"],roi["x1"]:roi["x2"]].max(axis=(1,2))
        roi_values.append({stage: _statistics(means[i,:,ri],maxima[i]) for i,stage in enumerate(STAGES)})
    return global_values, roi_values


def _flatten(stages, recurrence_applied):
    output = {stage+"_"+name: (value if recurrence_applied or stage not in ("previous_input","after_pda","after_tsfm") else None)
              for stage, stats in stages.items() for name,value in stats.items()}
    # Compatibility alias; these are source means, never a recurrent readout.
    output.update({"before_pda_"+name: output["previous_input_"+name]
                   for name in stages["previous_input"]})
    return output


def _logit_statistics(logits, row, roi=None):
    """Same ROI for all modes; report independent peaks and fixed-anchor values."""
    if roi is None:
        values = logits
    else:
        values = logits[:, roi["batch_index"], roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
    for mi, mode in enumerate(MODES):
        row[mode+"_z_mean"] = float(values[mi].mean(dtype=np.float64))
        row[mode+"_z_max"] = float(values[mi].max())
        row[mode+"_z_min"] = float(values[mi].min())
        if roi is not None:
            iy, ix = np.unravel_index(np.argmax(values[mi]), values[mi].shape)
            row[mode+"_peak_x"] = int(roi["x1"]+ix)
            row[mode+"_peak_y"] = int(roi["y1"]+iy)
            row[mode+"_anchor_z"] = float(logits[mi,roi["batch_index"],roi["anchor_y"],roi["anchor_x"]])
    difference = values[0]-values[1]
    row.update(delta_z_mean=float(difference.mean(dtype=np.float64)), delta_z_max=float(difference.max()),
               delta_z_min=float(difference.min()), delta_z_absmax=float(np.abs(difference).max()),
               history_current_pixel_flips=int(np.count_nonzero((values[0]>=0)!=(values[1]>=0))))
    if roi is not None:
        row["delta_z_anchor"] = row["History_anchor_z"]-row["HistoryCurrent_anchor_z"]
        row["peak_coordinates_changed"] = (row["History_peak_x"],row["History_peak_y"]) != (row["HistoryCurrent_peak_x"],row["HistoryCurrent_peak_y"])


def _read_with_observations(adapter, current, state, *, window_id, step, mode):
    """Observe reference-core pair operators without changing its bank update."""
    from revision import bank
    pairs = []

    def align(reference, now):
        aligned = adapter.align(reference, now)
        pairs.append((reference, aligned))
        return aligned

    def fuse(aligned, now):
        candidate = adapter.fuse(aligned, now)
        pairs[-1] = (*pairs[-1], candidate)
        return candidate

    readout, updated = bank.read_then_write(current, state, window_id=window_id,
        step_index=step, align=align, fuse=fuse, mode=mode)
    expected = list(range(max(0, step-adapter.capacity), step))
    assert [entry.step_index for entry in state.entries] == expected
    assert len(pairs) == len(expected)
    assert [entry.step_index for entry in updated.entries] == list(range(max(0, step+1-adapter.capacity), step+1))
    assert torch.equal(updated.entries[-1].feature, current)
    assert updated.entries[-1].feature.data_ptr() != current.data_ptr()
    return readout, updated, pairs


def _ordered_mean(values):
    """Match the source-order accumulation used by the FP32 reference core."""
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)


@torch.no_grad()
def trace_window(model, x, regions, *, sequence=None, start=0, frontend_core=128, frontend_halo=16):
    """Three same-weight modes; compact source-age evidence and fixed ROI probes.

    Every mode receives the same full-spatial F/H. History and current_only call
    the supplied read_then_write core independently, with the same 150 PDA/TSFM
    pairs. Readouts never enter the FIFO. Slot artifacts contain scalars only.
    """
    from revision import bank
    if model.training or model.adapter.training or model.base.training:
        raise ValueError("Trace requires eval() and frozen baseline BN")
    if x.ndim != 5 or x.shape[1:3] != (1,40):
        raise ValueError("Trace requires the original [B,1,40,H,W] window")
    if not isinstance(regions, Mapping) and len(regions) != 40:
        raise ValueError("regions must contain one ROI list per local frame")
    if frontend_core != 128 or frontend_halo != 16:
        raise ValueError("Use the registered frozen frontend core128/halo16")
    _validate_head(model.base)
    adapter = model.adapter
    assert adapter.state_channels == 16 and adapter.capacity == 4
    assert adapter.out_projection.kernel_size == (1,1)
    assert hasattr(adapter, "align") and hasattr(adapter, "fuse")
    assert not hasattr(adapter.recurrent_cell, "state_update")
    original_state = {name: value.detach().clone() for name,value in model.state_dict().items()}
    began = time.perf_counter()
    f, h = model.frozen_frontend_tiled(x, core=frontend_core, halo=frontend_halo)
    batch, _, length, height, width = f.shape
    logits = {mode: torch.empty((batch,length,height,width),dtype=x.dtype,device="cpu") for mode in MODES}
    frame_rows, roi_rows, slot_frame_rows = [], [], []
    identity_max = {"residual_projection":0.,"injected_H_projection":0.,"head_piecewise_secant":0.,
                    "bank_read_mean":0.}
    window_id = f"trace:{sequence}:{start}:ordered_rows:{tuple(range(batch))}"
    history_bank = bank.start_window(window_id, capacity=4)
    current_bank = bank.start_window(window_id, capacity=4)
    skipped_rois, total_pairs = 0, 0
    for t in range(length):
        current = adapter.in_projection(f[:,:,t])
        read_h, history_bank, pairs_h = _read_with_observations(adapter,current,history_bank,
            window_id=window_id,step=t,mode="history")
        read_c, current_bank, pairs_c = _read_with_observations(adapter,current,current_bank,
            window_id=window_id,step=t,mode="current_only")
        count = min(t,4)
        assert len(pairs_h) == len(pairs_c) == count
        total_pairs += count
        if count:
            source_h = _ordered_mean([value[0] for value in pairs_h])
            source_c = _ordered_mean([value[0] for value in pairs_c])
            aligned_h = _ordered_mean([value[1] for value in pairs_h])
            aligned_c = _ordered_mean([value[1] for value in pairs_c])
            identity_max["bank_read_mean"] = max(identity_max["bank_read_mean"],
                float((read_h-_ordered_mean([value[2] for value in pairs_h])).abs().max()),
                float((read_c-_ordered_mean([value[2] for value in pairs_c])).abs().max()))
            slot_maps = torch.stack([torch.stack([_moments(a,b) for a,b in zip(ph,pc)])
                                     for ph,pc in zip(pairs_h,pairs_c)])
            # Transfer only per-source scalar moments; no slot feature maps persist.
            slot_means = slot_maps[:,:,:4].mean(dim=(3,4,5)).cpu().numpy()
            slot_maxima = slot_maps[:,:,4].amax(dim=(2,3,4)).cpu().numpy()
            for slot, source_index in enumerate(range(t-count,t)):
                stats = {stage: _statistics(slot_means[slot,j],slot_maxima[slot,j])
                         for j,stage in enumerate(("source","after_pda","candidate"))}
                slot_frame_rows.append({"sequence":sequence,"window_start":int(start),"window_t":t,
                    "frame":int(start+t),"slot":slot,"source_window_t":source_index,
                    "source_frame":int(start+source_index),"source_age":t-source_index,
                    "active_slots":count,"current_control_actual_source_window_t":t,
                    **{stage+"_"+name:value for stage,values in stats.items() for name,value in values.items()}})
            del slot_maps,slot_means,slot_maxima
        else:
            source_h = source_c = aligned_h = aligned_c = current
        residual_h = adapter.out_projection(torch.cat((current,read_h),1))
        residual_c = adapter.out_projection(torch.cat((current,read_c),1))
        new_h, new_c = h[:,:,t]+residual_h, h[:,:,t]+residual_c
        head_h, head_c, head_off = (_head(model.base,v) for v in (new_h,new_c,h[:,:,t]))
        z = torch.stack((head_h[3][:,0],head_c[3][:,0],head_off[3][:,0]))
        z_cpu = z.cpu().numpy()
        for mi,mode in enumerate(MODES):
            logits[mode][:,t].copy_(torch.from_numpy(z_cpu[mi]))
        delta_m = read_h-read_c
        projected_delta = F.conv2d(delta_m,adapter.out_projection.weight[:,16:],bias=None)
        delta_h = new_h-new_c
        secant = F.conv2d(head_h[2]-head_c[2],model.base.conv_out2.weight[:,:,0],bias=None)
        identity = torch.stack(((residual_h-residual_c-projected_delta).abs().max(),
                                (delta_h-projected_delta).abs().max(),
                                (head_h[3]-head_c[3]-secant).abs().max())).cpu().tolist()
        for key,value in zip(("residual_projection","injected_H_projection","head_piecewise_secant"),identity):
            identity_max[key] = max(identity_max[key],float(value))
        pairs = ((source_h,source_c),(aligned_h,aligned_c),(read_h,read_c),
                 (residual_h,residual_c),(new_h,new_c),
                 (head_h[0],head_c[0]),(head_h[1],head_c[1]),(head_h[2],head_c[2]),(head_h[3],head_c[3]))
        maps = torch.stack([_moments(a,b) for a,b in pairs]).cpu().numpy()
        rois, skipped = _regions_at(regions,t,batch,height,width)
        skipped_rois += skipped
        global_stats, region_stats = _stage_statistics(maps,rois)
        shared = {"sequence":sequence,"window_start":int(start),"window_t":t,"frame":int(start+t),
                  "recurrence_applied":t>0,"bank_read_applied":t>0,"active_slots":count,
                  "source_indices":list(range(t-count,t)),"source_ages":list(range(count,0,-1))}
        row = dict(shared,**_flatten(global_stats,t>0))
        row.update({"identity_"+key+"_absmax":float(value) for key,value in zip(
            ("residual_projection","injected_H_projection","head_piecewise_secant"),identity)})
        _logit_statistics(z_cpu,row)
        frame_rows.append(row)
        if rois:
            bi, ay, ax = (torch.tensor([r[k] for r in rois],device=x.device,dtype=torch.long)
                          for k in ("batch_index","anchor_y","anchor_x"))
            dh_anchor = delta_h[bi,:,ay,ax]
            pre_h,pre_c,pre_o = [v[1][bi,:,ay,ax] for v in (head_h,head_c,head_off)]
            jacobians = [head_input_jacobian(model.base,v) for v in (pre_h,pre_c,pre_o)]
            anchor_info = torch.stack([value for jac in jacobians for value in
                                      (jac.norm(dim=1),(jac*dh_anchor).sum(1))]
                                     +[dh_anchor.norm(dim=1),((pre_h>0)!=(pre_c>0)).sum(1).to(x.dtype),
                                       (pre_h>0).sum(1).to(x.dtype),(pre_c>0).sum(1).to(x.dtype)]).cpu().numpy()
            for ri,(roi,stats) in enumerate(zip(rois,region_stats)):
                item = dict(roi,**shared,**_flatten(stats,t>0))
                _logit_statistics(z_cpu,item,roi)
                for mi,mode in enumerate(MODES):
                    norm,dot = map(float,anchor_info[2*mi:2*mi+2,ri])
                    dh_norm = float(anchor_info[6,ri])
                    item[mode+"_anchor_jacobian_l2"] = norm
                    item[mode+"_anchor_J_dot_deltaH"] = dot
                    item[mode+"_anchor_J_deltaH_cosine"] = dot/(norm*dh_norm) if norm*dh_norm>1e-12 else None
                    item[mode+"_anchor_linearization_error"] = item["delta_z_anchor"]-dot
                item.update(anchor_deltaH_l2=float(anchor_info[6,ri]),anchor_relu_flips=int(anchor_info[7,ri]),
                            History_anchor_active_relu=int(anchor_info[8,ri]),HistoryCurrent_anchor_active_relu=int(anchor_info[9,ri]))
                roi_rows.append(item)
        if t == 0:
            assert torch.equal(z[0],z[1]), "History and current_only must agree at t0"
            assert torch.equal(read_h,current) and torch.equal(read_c,current)
        del maps,pairs,pairs_h,pairs_c,head_h,head_c,head_off,z,z_cpu,delta_m,projected_delta,delta_h,secant
    assert total_pairs == 150
    assert all(torch.isfinite(value).all() for value in logits.values())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in original_state.items())
    return {"logits":logits,"frame_rows":frame_rows,"roi_rows":roi_rows,"slot_frame_rows":slot_frame_rows,
            "metadata":{"modes":list(MODES),"stages":list(STAGES),"input_shape":list(x.shape),
                        "maximum_identity_residual":identity_max,"elapsed_seconds":time.perf_counter()-began,
                        "global_m1_spatial_shape":[height,width],"shared_F_H":True,
                        "state_reset_at_window_start":True,"all_weights_and_buffers_unchanged":True,
                        "baseline_buffers_unchanged":True,
                        "pda_tsfm_calls_per_mode":{"History":total_pairs,"HistoryCurrent":total_pairs,"HistoryOff":0},
                        "bank_capacity":4,"bank_entries":"original projected X cloned before alignment/fusion; never readout",
                        "read_then_write":True,"first_frame":"readout=X0; zero source slots and no pair operators",
                        "previous_input_semantics":"source-order mean of original source X; current control substitutes current X in every slot",
                        "before_pda_semantics":"alias of previous_input for archived aggregation compatibility",
                        "after_pda_semantics":"source-order mean of separately aligned sources",
                        "after_tsfm_semantics":"actual core readout: source-order mean of separately fused candidates",
                        "slot_frame_statistics":"per-source full-frame scalar differences before PDA, after PDA and TSFM; no full feature maps persisted",
                        "empty_clipped_rois_skipped":skipped_rois,"roi_coordinate_convention":"integer half-open xyxy",
                        "delta_definition":"History minus same-checkpoint HistoryCurrent",
                        "HistoryOff_definition":"shared original H directly through the frozen head; no M1 residual",
                        "jacobian_definition":"local frozen-head logit derivative wrt H_new at fixed ROI anchor; PyTorch ReLU0 derivative0",
                        "roi_labels_enter_model":False}}


