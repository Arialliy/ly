"""Train-only adapters around the supplied reference ``visible_segments``.

All eight unchanged C-v1 windows contribute eligible annotated observations.
Visibility is absent in the frozen annotations: these are continuous legal
annotation proxies, not independently verified visible trajectories.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
V1 = ROOT / "experiments/track_contrast_v1"
REFERENCE = ROOT / "deeppro_c_v2_reference/segments.py"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _module("cv2_supplied_segments", REFERENCE)
old = _module("cv2_original_pair_sampler", V1 / "train_modules/track_pair_sampler.py")
Observation = reference.Observation
visible_segments = reference.visible_segments

CONFIG = {
    "length": 8, "stride": 4, "source_frame_step": 1,
    "tail_alignment": True, "full_length_only": True,
    "target_max_side": 16, "core_margin": 9, "window_length": 40,
    "windows_per_entry": 8, "entries": 256,
    "identity_rule": "unchanged_v1_video_scoped_identity_exclusions_including34_ambiguous_tracks",
    "visibility_rule": "continuous_legal_annotation_proxy;source_visibility_fields_absent;no_interpolation",
    "target_pool": "all_eligible_observations_in_all8_windows;not_v1_common_B_C_subset",
    "background_rule": "release_teacher;all_annotation_exclusion;full_ROI_legal;local32_and_global;at_least_one_required",
    "global_window_mapping": "primary0..3_to_aux4..7;aux4..7_to_itself",
    "background_gap_rule": "no_available_negative_breaks_run;rebuild_full_segments",
    "reduction": "mean_segments_per_window_then_fixed8_window_mean",
}


def continuous_runs(descriptors, valid=None):
    """Split both input/source-frame gaps, absent rows and explicit invalidity.

    The window index belongs to the group key, so two overlapping exposures
    never merge. Original frame and source frame must both advance by one.
    """
    if valid is None:
        valid = [True] * len(descriptors)
    if len(valid) != len(descriptors):
        raise ValueError("valid mask must have one entry per descriptor")
    groups = defaultdict(list)
    seen = set()
    for row, descriptor in enumerate(descriptors):
        key = (descriptor["window"], descriptor["sequence"],
               descriptor["video_id"], descriptor["track_id"])
        unique = key + (descriptor["frame"],)
        if unique in seen:
            raise ValueError("duplicate video-scoped track/frame/window")
        seen.add(unique)
        groups[key].append(row)
    runs = []
    for key in sorted(groups):
        run = []
        for row in sorted(groups[key], key=lambda r: descriptors[r]["frame"]):
            item = descriptors[row]
            if not bool(valid[row]):
                if run:
                    runs.append(run)
                run = []
                continue
            if run:
                previous = descriptors[run[-1]]
                if (item["frame"] != previous["frame"] + 1
                    or item["source_frame"] != previous["source_frame"] + 1
                    or item["slot"] != previous["slot"] + 1):
                    runs.append(run)
                    run = []
            run.append(row)
        if run:
            runs.append(run)
    return runs


def build_segments(descriptors, valid=None, *, length=8, stride=4, tail_alignment=True):
    """Return full row-index segments using the supplied reference core.

    The sole sampling extension is one final aligned segment when a run's
    remainder would otherwise leave legal observations uncovered.
    """
    if length < 2 or stride < 1:
        raise ValueError("length>=2 and stride>=1 required")
    runs = continuous_runs(descriptors, valid)
    segments, windows = [], []
    before_tail, covered = set(), set()
    tail_count = 0
    for run_number, run in enumerate(runs):
        if len(run) < length:
            continue
        first = descriptors[run[0]]
        video = f"{first['sequence']}:{first['video_id']}:w{first['window']}:run{run_number}"
        observations = [Observation(video=video, track=str(first["track_id"]),
            frame=descriptors[row]["source_frame"], row=row,
            visible_and_labeled=True) for row in run]
        # True here means legal annotation proxy; it is not a visibility claim.
        chosen = visible_segments(observations, length=length, step=1, stride=stride)
        before_tail.update(row for segment in chosen for row in segment)
        if tail_alignment and chosen[-1][-1] != run[-1]:
            tail = visible_segments(observations[-length:], length=length, step=1, stride=stride)
            assert len(tail) == 1 and tail[0] not in chosen
            chosen.extend(tail)
            tail_count += 1
        segments.extend([list(s) for s in chosen])
        windows.extend([first["window"]] * len(chosen))
        covered.update(row for segment in chosen for row in segment)
    histogram = Counter(len(run) for run in runs)
    return {
        "segment_indices": segments, "segment_windows": windows,
        "coverage": {"runs": len(runs), "run_length_histogram": dict(sorted(histogram.items())),
            "valid_observations": sum(map(len, runs)), "segments": len(segments),
            "covered_observations": len(covered), "tail_segments": tail_count,
            "tail_extra_observations": len(covered - before_tail),
            "segment_frame_exposures": len(segments) * length},
    }


class SegmentSampler:
    def __init__(self):
        self.sampler = old.TrackPairSampler()
        self.sequences = self.sampler.sequences
        self.v1_audit = json.loads((V1 / "SAMPLER_AUDIT.json").read_text(encoding="utf-8"))
        expected = {value["sequence"]: value for value in self.v1_audit["sequence_audits"]}
        for name, sequence in self.sequences.items():
            assert sequence.audit["annotation_sha256"] == expected[name]["annotation_sha256"]
            assert sequence.audit["excluded_uncertain_track_ids"] == expected[name]["excluded_uncertain_track_ids"]
        assert sum(s.audit["uncertain_tracks"] for s in self.sequences.values()) == 34

    def sample(self, entry, auxiliary):
        descriptors = []
        window = 0
        for exposure in (entry, auxiliary):
            for pair in exposure["pairs"]:
                sequence = self.sequences[pair["sequence"]]
                crop = pair["crop_yx"]
                for start in pair["starts"]:
                    for frame in range(start, start + 40):
                        for track, annotation in sorted(sequence.eligible[frame].items()):
                            if old.full_core(annotation, crop):
                                value = sequence.descriptor(annotation, frame, window, start, crop)
                                value["eligibility_basis"] = "continuous_legal_annotation_proxy"
                                descriptors.append(value)
                    window += 1
        assert window == 8
        segment_plan = build_segments(descriptors, length=CONFIG["length"],
            stride=CONFIG["stride"], tail_alignment=CONFIG["tail_alignment"])
        result = {"index": entry["step"], "target_regions": [
            [d["window"], d["slot"], *d["rect"]] for d in descriptors],
            "descriptors": descriptors, **segment_plan}
        result["plan_sha256"] = old.content_hash(result)
        return result
