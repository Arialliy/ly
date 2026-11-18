"""Full-frame scoring and authenticated release comparisons for DJI_0067_2."""
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE = "DJI_0067_2"
PILOT = ROOT / "sources/temporal_diagnosis_20261010/stage1_goal/pilot/score_pilot.py"
ARCHIVE = (ROOT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010"
           / "stage1_goal/predictions/official" / SEQUENCE)
SCORES = ROOT / "results/references/release_evaluation" / SEQUENCE
COUNT_FIELDS = ("gt_count", "matched_count", "component_count", "unmatched_components",
                "positive_pixels", "positive_pixels_outside_boxes")


class ReleaseEvaluation:
    """Load requested release frames and check both window fusion and historical scores."""

    official_starts = (0, 36, 72, 108, 120)
    threshold = 0.5
    shape = (512, 640)

    def __init__(self, frames):
        frames = tuple(sorted(set(frames)))
        if not frames or any(not isinstance(f, (int, np.integer)) or not 0 <= f < 160
                             for f in frames):
            raise ValueError("Pass one or more zero-based frame indices in [0, 159]")
        frames = tuple(map(int, frames))
        spec = importlib.util.spec_from_file_location("release_score_pilot", PILOT)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        self._matcher, matcher_path = pilot.helper()
        score_manifest = json.loads((SCORES / "COMPLETE.json").read_text())
        prediction_manifest = json.loads((ARCHIVE / "COMPLETE.json").read_text())
        if (score_manifest["sequence"] != SEQUENCE or score_manifest["condition"] != "release"
                or score_manifest["threshold"] != self.threshold
                or prediction_manifest["sequence"] != SEQUENCE
                or tuple(prediction_manifest["official_starts"]) != self.official_starts
                or any(m["status"] != "COMPLETE" for m in (score_manifest, prediction_manifest))):
            raise ValueError("Unexpected historical condition, threshold or fusion schedule")
        sources = {Path(k).name: v for k, v in score_manifest["source_sha256"].items()}
        annotations = ROOT / "data/stage1" / SEQUENCE / "annotations.json"
        for path in (annotations, matcher_path, ARCHIVE / "COMPLETE.json"):
            if pilot.digest(path) != sources[path.name]:
                raise ValueError(f"Historical source digest mismatch for {path}")
        all_gt, _, offsets, _ = pilot.gt_data(annotations, self._matcher)
        self.ground_truth = {frame: all_gt[frame] for frame in frames}
        self.fused = {}
        self.windows = {}
        fused_path = ARCHIVE / "official_fused.npz"
        if not pilot.digest(fused_path) == prediction_manifest["fused_sha256"] == sources[fused_path.name]:
            raise ValueError("Historical fused archive digest mismatch")
        with np.load(fused_path, allow_pickle=False) as archive:
            if tuple(archive["starts"]) != self.official_starts:
                raise ValueError("Historical fused archive declares different windows")
            probabilities = archive["probabilities"]
            for frame in frames:
                field = probabilities[frame].copy()
                field.setflags(write=False)
                self.fused[frame] = field
            del probabilities
        for start in self.official_starts:
            covered = [frame for frame in frames if start in self.covering_starts(frame)]
            if not covered:
                continue
            path = ARCHIVE / "windows" / f"window_{start:03d}.npz"
            if not pilot.digest(path) == prediction_manifest["window_sha256"][path.name] == sources[path.name]:
                raise ValueError(f"Historical window archive digest mismatch for {path}")
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["start"]) != start or int(archive["end"]) != start + 40:
                    raise ValueError(f"Historical window indices differ in {path}")
                probabilities = archive["probabilities"]
                for frame in covered:
                    field = probabilities[frame - start].copy()
                    field.setflags(write=False)
                    self.windows[start, frame] = field
                del probabilities
        self.baseline = {}
        with np.load(SCORES / "scores.npz", allow_pickle=False) as archive:
            plan, = np.flatnonzero(archive["plans"] == "O")
            if float(archive["threshold"]) != self.threshold or not np.array_equal(archive["offsets"], offsets):
                raise ValueError("Frozen score threshold or GT ordering differs")
            counts = archive["frame_counts"][plan]
            matched = archive["matched"][plan]
            maxima = archive["box_max"][plan]
            for frame in frames:
                fused = np.maximum.reduce([self.windows[start, frame] for start in self.covering_starts(frame)])
                np.testing.assert_array_equal(fused, self.fused[frame])
                rows, stats = self.score(frame, self.fused[frame])
                lo, hi = offsets[frame:frame + 2]
                metadata = score_manifest["objects"][lo:hi]
                keys = [(row["annotation_id"], row["track_id"]) for row in rows.values()]
                expected_keys = [(obj["annotation_id"], obj["track_id"]) for obj in metadata]
                if keys != expected_keys or any(obj["frame"] != frame for obj in metadata):
                    raise ValueError(f"Ground-truth identity/order mismatch in frame {frame}")
                np.testing.assert_array_equal([row["matched"] for row in rows.values()], matched[lo:hi])
                np.testing.assert_array_equal(
                    np.array([row["box_max"] for row in rows.values()], dtype=np.float32), maxima[lo:hi])
                np.testing.assert_array_equal([stats[key] for key in COUNT_FIELDS], counts[frame])
                self.baseline[frame] = (rows, stats)
        self.verification = {
            "sequence": SEQUENCE, "threshold": self.threshold, "frames": list(frames),
            "object_frames_verified": sum(len(rows) for rows, _ in self.baseline.values()),
            "window_frame_maps_verified": len(self.windows),
            "fused_sha256": prediction_manifest["fused_sha256"],
            "annotations_sha256": sources["annotations.json"],
            "matching_source_sha256": sources["analyze.py"],
            "exact_checks": ["official window maximum", "annotation ID and track ID ordering",
                             "per-object matched and box_max", *COUNT_FIELDS],
        }

    @classmethod
    def covering_starts(cls, frame):
        return tuple(start for start in cls.official_starts if start <= frame < start + 40)

    def score(self, frame, probability):
        """Return ({annotation_id: release row}, frame-level release diagnostics)."""
        if frame not in self.ground_truth:
            raise KeyError(f"Frame {frame} was not requested")
        probability = np.asarray(probability)
        if (probability.shape != self.shape or not np.isfinite(probability).all()
                or probability.min() < 0 or probability.max() > 1):
            raise ValueError("Expected a finite full-frame 512 x 640 probability map in [0,1]")
        rows, stats, _ = self._matcher.match_frame(probability, self.ground_truth[frame], self.threshold)
        return {row["annotation_id"]: row for row in rows}, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=int, help="Zero-based endpoint frames to replay")
    evaluation = ReleaseEvaluation(parser.parse_args().frames)
    print(json.dumps(evaluation.verification, indent=2))
