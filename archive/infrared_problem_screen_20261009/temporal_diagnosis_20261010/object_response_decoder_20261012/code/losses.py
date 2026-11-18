"""Direct object-response supervision with frame or whole-track assignment.

All reductions are shared by R0/R1/R2.  Only the assignment constraint changes.
The loss consumes the sparse, annotation-only dictionaries made by targets.py.
There is never a [queries, objects, height, width] tensor.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


LOSS_CONFIG = {
    "version": 1,
    "terms": ["existence_positive", "existence_negative", "mask_positive",
              "mask_instance_negative", "mask_background_negative"],
    "coefficients": [1.0] * 5,
    "reduction": "five independent means per window; caller averages four windows",
    "matching_cost": "existence_positive + mask_positive + eligible_mask_instance_negative",
    "track_matching_reduction": "mean over each track's valid observations only",
    "loss_positive_reduction": "object-frame mean in both assignment modes",
    "matching_solver": "scipy.optimize.linear_sum_assignment CPU float64",
    "unknown_rule": "shared targets.negative_eligible; no negative terms otherwise",
    "mask_sum_dtype": "float32; same precision in all three configurations",
}


def _key(value):
    key = tuple(value)
    if len(key) != 3:
        raise ValueError("Expected canonical (sequence, category_id, str(tracking_id))")
    return str(key[0]), int(key[1]), str(key[2])


def _sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _validate(a, masks, targets):
    if a.ndim != 2 or masks.ndim != 4 or masks.shape[:2] != a.shape:
        raise ValueError("Expected a[T,K] and masks[T,K,H,W]")
    if a.device != masks.device or a.dtype != torch.float32 or masks.dtype != torch.float32:
        raise ValueError("All direct logits must be FP32 on one device")
    t, k, h, w = masks.shape
    if not t or not k or not h or not w:
        raise ValueError("Empty logit dimensions")
    valid = np.asarray(targets["valid"])
    background = np.asarray(targets["background"])
    eligible = np.asarray(targets["negative_eligible"])
    if (valid.shape != (t, h, w) or background.shape != valid.shape
            or eligible.shape != (t,) or valid.dtype != np.bool_
            or background.dtype != np.bool_ or eligible.dtype != np.bool_):
        raise ValueError("Invalid valid/background/negative_eligible target arrays")
    if np.any(background & ~valid):
        raise ValueError("Physical background must be a subset of valid pixels")
    observations = sorted(targets["observations"],
                          key=lambda o: (int(o["frame"]), _key(o["track_key"]),
                                         int(o["annotation_id"])))
    tracks = sorted({_key(o["track_key"]) for o in observations})
    if "track_keys" in targets and tracks != sorted(_key(v) for v in targets["track_keys"]):
        raise ValueError("track_keys disagree with valid observations")
    seen = set()
    frame_counts = np.zeros(t, dtype=np.int64)
    for obs in observations:
        frame = int(obs["frame"])
        identity = _key(obs["track_key"])
        if not 0 <= frame < t or (frame, identity) in seen:
            raise ValueError("Invalid frame or duplicate same-frame identity")
        seen.add((frame, identity))
        support = np.asarray(obs["support_indices"])
        if (support.ndim != 1 or support.dtype != np.int64 or not len(support)
                or np.any(support < 0) or np.any(support >= h*w)
                or len(np.unique(support)) != len(support)
                or not np.all(valid[frame].ravel()[support])):
            raise ValueError("Object support must be nonempty, unique, and inside valid domain")
        if np.any(background[frame].ravel()[support]):
            raise ValueError("Annotated target support entered physical background")
        frame_counts[frame] += 1
    if np.any(frame_counts > k):
        raise ValueError("More frame objects than queries; truncating GT is forbidden")
    # This makes no-object term counts identical for frame and track assignments.
    if np.any(eligible & (frame_counts != len(tracks))):
        raise ValueError("Eligible frame must observe every valid window track")
    return valid, background, eligible, observations, tracks


def _assign(cost, observations, tracks, query_count, mode):
    """One CPU cost transfer upstream; all Hungarian calls remain on the CPU."""
    if mode not in ("frame", "track"):
        raise ValueError("assignment must be frame or track")
    if not np.all(np.isfinite(cost)):
        raise FloatingPointError("Non-finite Hungarian costs")
    selected = np.full(len(observations), -1, dtype=np.int64)
    if mode == "frame":
        groups = {}
        for index, obs in enumerate(observations):
            groups.setdefault(int(obs["frame"]), []).append(index)
        for indices in groups.values():
            rows, cols = linear_sum_assignment(cost[np.asarray(indices)])
            selected[np.asarray(indices)[rows]] = cols
    else:
        if len(tracks) > query_count:
            raise ValueError("More whole-window tracks than queries; truncating GT is forbidden")
        indices_by_track = {track: [] for track in tracks}
        for index, obs in enumerate(observations):
            indices_by_track[_key(obs["track_key"])].append(index)
        if tracks:
            track_cost = np.stack([cost[indices_by_track[key]].mean(axis=0)
                                   for key in tracks])
            rows, cols = linear_sum_assignment(track_cost)
            for row, col in zip(rows, cols):
                selected[indices_by_track[tracks[row]]] = int(col)
    if np.any(selected < 0):
        raise AssertionError("A valid object observation was not assigned")
    return selected


def decoder_loss(a, masks, targets, assignment="frame"):
    """Return ``(loss, metrics, assignments)`` for one T-frame training window.

    The scalar retains gradients.  metrics contain small JSON-safe scalars and
    assignment/cost hashes; assignments are returned for finite semantic checks,
    not meant to be copied into every formal-training log.
    """
    valid_np, bg_np, eligible_np, observations, tracks = _validate(a, masks, targets)
    t, k, h, w = masks.shape
    device = masks.device
    valid_count_np = valid_np.sum(axis=(1, 2), dtype=np.int64)
    bg_count_np = bg_np.sum(axis=(1, 2), dtype=np.int64)
    valid = torch.as_tensor(valid_np, device=device)
    bg = torch.as_tensor(bg_np, device=device)
    valid_count = torch.as_tensor(valid_count_np, device=device, dtype=torch.float32)
    bg_count = torch.as_tensor(bg_count_np, device=device, dtype=torch.float32)

    # Three [T,K] tables summarize the full spatial negative loss.  No object
    # dimension is introduced here.  torch.where avoids a large float mask copy.
    negative = F.softplus(masks)
    valid_sum = torch.where(valid[:, None], negative, 0.0).sum(dim=(-1, -2))
    bg_sum = torch.where(bg[:, None], negative, 0.0).sum(dim=(-1, -2))
    bg_mean = bg_sum / bg_count.clamp_min(1)[:, None]
    # Zero remains connected to both independent output heads, including an
    # entirely unknown synthetic window, without another full spatial reduction.
    zero = a.reshape(-1)[0] * 0.0 + masks.reshape(-1)[0] * 0.0
    n = len(observations)
    frame_np = np.asarray([int(o["frame"]) for o in observations], dtype=np.int64)
    frame = torch.as_tensor(frame_np, device=device)
    if n:
        lengths_np = np.asarray([len(o["support_indices"]) for o in observations], dtype=np.int64)
        maximum = int(lengths_np.max())
        indices_np = np.zeros((n, maximum), dtype=np.int64)
        support_valid_np = np.zeros((n, maximum), dtype=np.bool_)
        for oi, obs in enumerate(observations):
            length = len(obs["support_indices"])
            indices_np[oi, :length] = obs["support_indices"]
            support_valid_np[oi, :length] = True
        indices = torch.as_tensor(indices_np, device=device)
        support_valid = torch.as_tensor(support_valid_np, device=device)
        lengths = torch.as_tensor(lengths_np, device=device, dtype=torch.float32)
        queries = torch.arange(k, device=device)
        # Sparse padded supports [objects,K,max_support].  Padding logits may be
        # read but their values and gradients are removed by the explicit mask.
        gathered = masks.flatten(2)[frame[:, None, None], queries[None, :, None],
                                   indices[:, None, :]]
        positive_mean = torch.where(support_valid[:, None], F.softplus(-gathered),
                                    0.0).sum(dim=-1) / lengths[:, None]
        support_negative_sum = torch.where(support_valid[:, None], F.softplus(gathered),
                                           0.0).sum(dim=-1)
        instance_counts = valid_count[frame] - lengths
        instance_mean = ((valid_sum[frame] - support_negative_sum)
                         / instance_counts.clamp_min(1)[:, None])
        instance_valid_np = eligible_np[frame_np] & ((valid_count_np[frame_np]-lengths_np) > 0)
        instance_valid = torch.as_tensor(instance_valid_np, device=device)
        with torch.no_grad():
            cost = (F.softplus(-a[frame]) + positive_mean.detach()
                    + torch.where(instance_valid[:, None], instance_mean.detach(), 0.0))
            # One transfer for every frame/track cost in the legal window.
            cost_np = cost.to(device="cpu", dtype=torch.float64).numpy()
        selected_np = _assign(cost_np, observations, tracks, k, assignment)
        selected = torch.as_tensor(selected_np, device=device)
        rows = torch.arange(n, device=device)
        existence_positive = F.softplus(-a[frame, selected]).mean()
        mask_positive = positive_mean[rows, selected].mean()
        chosen_instance = instance_mean[rows, selected]
        mask_instance_negative = (chosen_instance[instance_valid].mean()
                                  if np.any(instance_valid_np) else zero)
    else:
        if assignment not in ("frame", "track"):
            raise ValueError("assignment must be frame or track")
        cost_np = np.empty((0, k), dtype=np.float64)
        selected_np = np.empty(0, dtype=np.int64)
        instance_valid_np = np.empty(0, dtype=np.bool_)
        existence_positive = mask_positive = mask_instance_negative = zero

    negative_queries_np = np.broadcast_to(eligible_np[:, None], (t, k)).copy()
    negative_queries_np[frame_np, selected_np] = False
    negative_queries = torch.as_tensor(negative_queries_np, device=device)
    negative_background_np = negative_queries_np & (bg_count_np[:, None] > 0)
    negative_background = torch.as_tensor(negative_background_np, device=device)
    existence_negative = (F.softplus(a[negative_queries]).mean()
                          if np.any(negative_queries_np) else zero)
    mask_background_negative = (bg_mean[negative_background].mean()
                                if np.any(negative_background_np) else zero)
    terms = torch.stack([existence_positive, existence_negative, mask_positive,
                         mask_instance_negative, mask_background_negative])
    total = terms.sum()
    assignments = [{"frame": int(o["frame"]), "track_key": list(_key(o["track_key"])),
                    "annotation_id": int(o["annotation_id"]), "query": int(q)}
                   for o, q in zip(observations, selected_np)]
    # A second small transfer contains only the five requested loss metrics.
    term_values = terms.detach().to("cpu").tolist()
    if not np.all(np.isfinite(term_values)):
        raise FloatingPointError("Non-finite direct-supervision loss")
    metrics = dict(zip(LOSS_CONFIG["terms"], map(float, term_values)))
    metrics.update({
        "loss": float(sum(term_values)), "assignment": assignment,
        "object_frames": n, "tracks": len(tracks), "queries": k,
        "negative_eligible_frames": int(np.count_nonzero(eligible_np)),
        "existence_negative_query_frames": int(np.count_nonzero(negative_queries_np)),
        "instance_negative_object_frames": int(np.count_nonzero(instance_valid_np)),
        "background_negative_query_frames": int(np.count_nonzero(negative_background_np)),
        "empty_background_frames": int(np.count_nonzero(bg_count_np == 0)),
        "assignment_sha256": _sha(assignments),
        "cost_sha256": hashlib.sha256(cost_np.astype("<f8", copy=False).tobytes()).hexdigest(),
        "cost_shape": list(cost_np.shape),
    })
    return total, metrics, assignments


loss = decoder_loss
object_response_loss = decoder_loss
