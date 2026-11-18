"""M1-v1.2 same-weight traces including bounded state update and retention.

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
STAGES = ("before_pda", "after_pda", "after_tsfm", "after_update", "residual", "injected_H",
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
    return {stage+"_"+name: (value if recurrence_applied or stage not in ("before_pda","after_pda","after_tsfm") else None)
            for stage, stats in stages.items() for name,value in stats.items()}


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


def retention_statistics(gates, roi=None):
    """Scalar old-state retention distributions, with no invented gate at t0."""
    names = ("History", "HistoryCurrent")
    stats = ("mean", "std", "min", "q05", "q50", "q95", "max", "fraction_lt005", "fraction_gt095", "anchor")
    output = {"retention_" + mode + "_" + stat: None for mode in names for stat in stats}
    output.update(retention_delta_mean=None, retention_delta_rms=None, retention_delta_absolute_mean=None)
    if gates is None:
        return output
    assert gates.ndim == 4 and gates.shape[0] == 2
    assert np.isfinite(gates).all() and gates.min() >= 0 and gates.max() <= 1
    values = gates if roi is None else gates[:,roi["batch_index"],roi["y1"]:roi["y2"],roi["x1"]:roi["x2"]]
    for index, mode in enumerate(names):
        value = values[index].astype(np.float64, copy=False)
        q05,q50,q95 = np.quantile(value, [.05,.5,.95])
        result = dict(mean=float(value.mean()), std=float(value.std()), min=float(value.min()),
                      q05=float(q05), q50=float(q50), q95=float(q95), max=float(value.max()),
                      fraction_lt005=float((value < .05).mean()), fraction_gt095=float((value > .95).mean()),
                      anchor=None if roi is None else float(gates[index,roi["batch_index"],roi["anchor_y"],roi["anchor_x"]]))
        output.update({"retention_"+mode+"_"+key:val for key,val in result.items()})
    delta = values[0].astype(np.float64)-values[1].astype(np.float64)
    output.update(retention_delta_mean=float(delta.mean()), retention_delta_rms=float(np.sqrt(np.square(delta).mean())),
                  retention_delta_absolute_mean=float(np.abs(delta).mean()))
    return output


@torch.no_grad()
def trace_window(model, x, regions, *, sequence=None, start=0, frontend_core=128, frontend_halo=16):
    """Return three CPU logit windows, per-frame/ROI rows and identity evidence.

    regions is len40 or {local_frame: [ROI,...]}. ROI fields: id, kind,
    integer x1/y1/x2/y2 (half-open), groups; optional pair_id, arrangement,
    batch_index and frozen anchor_x/anchor_y. Coordinates never affect inference.
    """
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
    assert adapter.state_channels == 16
    assert adapter.out_projection.kernel_size == (1,1)
    cell = adapter.recurrent_cell
    if not all(hasattr(cell, name) for name in ("pda", "tsfm", "state_update")):
        raise TypeError("Trace requires actual PDA/TSFM and the v1.2 state_update")
    original_state = {name: value.detach().clone() for name,value in model.state_dict().items()}
    began = time.perf_counter()
    f, h = model.frozen_frontend_tiled(x, core=frontend_core, halo=frontend_halo)
    batch, _, length, height, width = f.shape
    logits = {mode: torch.empty((batch,length,height,width),dtype=x.dtype,device="cpu") for mode in MODES}
    frame_rows, roi_rows = [], []
    identity_max = {"residual_projection":0.,"injected_H_projection":0.,"head_piecewise_secant":0.}
    state = None
    previous_delta = None
    skipped_rois = 0
    for t in range(length):
        current = adapter.in_projection(f[:,:,t])
        if t == 0:
            previous = aligned_history = aligned_current = current
            state = state_current = candidate_history = candidate_current = current
            gate_history = gate_current = None
        else:
            previous = state
            aligned_history = cell.pda(previous.contiguous(),current.contiguous())
            aligned_current = cell.pda(current.contiguous(),current.contiguous())
            candidate_history = cell.tsfm(aligned_history,current)
            candidate_current = cell.tsfm(aligned_current,current)
            state, gate_history = cell.state_update(aligned_history,candidate_history,current,scope="pixel")
            state_current, gate_current = cell.state_update(aligned_current,candidate_current,current,scope="pixel")
        residual_h = adapter.out_projection(torch.cat((current,state),1))
        residual_c = adapter.out_projection(torch.cat((current,state_current),1))
        new_h, new_c = h[:,:,t]+residual_h, h[:,:,t]+residual_c
        head_h, head_c, head_off = (_head(model.base,v) for v in (new_h,new_c,h[:,:,t]))
        z = torch.stack((head_h[3][:,0],head_c[3][:,0],head_off[3][:,0]))
        z_cpu = z.cpu().numpy()
        for mi,mode in enumerate(MODES):
            logits[mode][:,t].copy_(torch.from_numpy(z_cpu[mi]))
        delta_m = state-state_current
        projected_delta = F.conv2d(delta_m,adapter.out_projection.weight[:,16:],bias=None)
        delta_h = new_h-new_c
        secant = F.conv2d(head_h[2]-head_c[2],model.base.conv_out2.weight[:,:,0],bias=None)
        identity = torch.stack(((residual_h-residual_c-projected_delta).abs().max(),
                                (delta_h-projected_delta).abs().max(),
                                (head_h[3]-head_c[3]-secant).abs().max())).cpu().tolist()
        for key,value in zip(identity_max,identity):
            identity_max[key] = max(identity_max[key],float(value))
        pairs = ((previous,current),(aligned_history,aligned_current),(candidate_history,candidate_current),(state,state_current),
                 (residual_h,residual_c),(new_h,new_c),
                 (head_h[0],head_c[0]),(head_h[1],head_c[1]),(head_h[2],head_c[2]),(head_h[3],head_c[3]))
        maps = torch.stack([_moments(a,b) for a,b in pairs]).cpu().numpy()
        rois, skipped = _regions_at(regions,t,batch,height,width)
        skipped_rois += skipped
        global_stats, region_stats = _stage_statistics(maps,rois)
        shared = {"sequence":sequence,"window_start":int(start),"window_t":t,"frame":int(start+t),
                  "recurrence_applied":t>0}
        row = dict(shared,**_flatten(global_stats,t>0))
        delta = global_stats["after_update"]["delta_rms"]
        row["delta_M_rms_ratio_previous"] = delta/previous_delta if previous_delta is not None and previous_delta>1e-12 else None
        previous_delta = delta
        row.update({"identity_"+key+"_absmax":float(value) for key,value in zip(identity_max,identity)})
        _logit_statistics(z_cpu,row)
        gates = None if t == 0 else torch.stack((gate_history[:,0],gate_current[:,0])).cpu().numpy()
        row.update(retention_statistics(gates))
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
                item.update(retention_statistics(gates,roi))
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
        del maps,pairs,head_h,head_c,head_off,z,z_cpu,delta_m,projected_delta,delta_h,secant
    assert all(torch.isfinite(value).all() for value in logits.values())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in original_state.items())
    return {"logits":logits,"frame_rows":frame_rows,"roi_rows":roi_rows,
            "metadata":{"modes":list(MODES),"stages":list(STAGES),"input_shape":list(x.shape),
                        "maximum_identity_residual":identity_max,"elapsed_seconds":time.perf_counter()-began,
                        "global_m1_spatial_shape":[height,width],"shared_F_H":True,
                        "state_reset_at_window_start":True,"all_weights_and_buffers_unchanged":True,
                        "baseline_buffers_unchanged":True,
                        "pda_tsfm_calls_per_mode":{"History":length-1,"HistoryCurrent":length-1,"HistoryOff":0},
                        "state_update_calls_per_mode":{"History":length-1,"HistoryCurrent":length-1,"HistoryOff":0},
                        "state_update_scope":"pixel",
                        "after_tsfm_semantics":"bare TSFM candidate before bounded state update",
                        "after_update_semantics":"(1-retention)*candidate+retention*aligned; actual recursive state",
                        "retention_semantics":"old aligned-state retention coefficient, not target probability or calibrated uncertainty; unused and null at t0",
                        "empty_clipped_rois_skipped":skipped_rois,"roi_coordinate_convention":"integer half-open xyxy",
                        "delta_definition":"History minus same-checkpoint HistoryCurrent",
                        "HistoryOff_definition":"shared original H directly through the frozen head; no M1 residual",
                        "temporal_ratio_interpretation":"Descriptive adjacent-frame difference ratio with changing inputs; not causal memory half-life",
                        "jacobian_definition":"local frozen-head logit derivative wrt H_new at fixed ROI anchor; PyTorch ReLU0 derivative0",
                        "roi_labels_enter_model":False}}
