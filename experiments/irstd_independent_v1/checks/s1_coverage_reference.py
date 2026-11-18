"""Independent check of S1's training-window fit and post-selection path diagnostic."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def _component(path):
    with Image.open(path) as mask:
        binary = np.asarray(mask) == 255
    if not binary.any():
        return None
    _, count = label(binary, structure=np.ones((3, 3), dtype=bool))
    if count != 1:
        return None
    y, x = np.nonzero(binary)
    centroid = (x.mean(), y.mean())
    # Pixel centers are integers; boxes include each annotated pixel's full extent.
    bounds = (x.min() - 0.5, y.min() - 0.5, x.max() + 0.5, y.max() + 0.5)
    return centroid, bounds


def verify_grid_reference(project_root: Path, experiment: Path, evidence: dict) -> dict:
    """Use a separate decoder, component labeler, fitting method and grid enumerator."""
    manifest = json.loads((experiment / "manifests/nudt_mirsdt_mask17.json").read_bytes())
    contexts = np.array([*range(-8, 0), *range(1, 9)])
    slopes, extents, centers, context_boxes = [], [], [], []
    total = 0
    for source in manifest["sequences"]:
        if source["split"] != "train":
            continue
        components = {row["number"]: _component(project_root / row["mask"]) for row in source["frames"]}
        for center in source["centers"]:
            total += 1
            current = components[center]
            neighbours = [components[center + int(dt)] for dt in contexts]
            if current is None or any(item is None for item in neighbours):
                continue
            xy = np.array(current[0])
            bounds = np.array(current[1])
            shifts = np.array([item[0] for item in neighbours]) - xy
            slope, _, _, _ = np.linalg.lstsq(contexts[:, None], shifts, rcond=None)
            slopes.append(slope[0])
            centers.append(xy)
            extents.append(bounds[2:] - bounds[:2])
            context_boxes.append([item[1] for item in neighbours])
    if (total, len(slopes)) != (evidence["counts"]["all_centers"], evidence["counts"]["fitted"]):
        raise ValueError("Independent reference and primary calculation use different window counts")
    if not slopes:
        raise ValueError("Formal reference requires eligible single-component training windows")
    slopes, extents = np.array(slopes), np.array(extents)
    selected = evidence["selected"]
    chosen = None
    counts = []
    for row in evidence["candidates"]:
        radius, step = row["radius"], row["step"]
        axis = np.linspace(-radius, radius, round(2 * radius / step) + 1)
        # Directly enumerate distances instead of deriving the nearest lattice index.
        distances = np.abs(slopes[..., None] - axis)
        errors = distances.min(axis=-1)
        in_range = int((np.abs(slopes) <= radius).all(axis=1).sum())
        accurate = int((8 * errors <= extents / 2).all(axis=1).sum())
        for count, key in ((in_range, "range_fraction"), (accurate, "quantization_fraction")):
            # A different count (one window) is readily distinguishable from float roundoff.
            if not np.isclose(row[key], count / len(slopes), atol=1e-12, rtol=0):
                raise ValueError(f"Independent {key} count differs for radius {radius}, step {step}")
        counts.append({"radius": radius, "step": step, "in_range": in_range, "within_quantization_limit": accurate})
        if selected and (radius, step) == (selected["radius"], selected["step"]):
            chosen = axis[distances.argmin(axis=-1)]
    if chosen is None:
        raise ValueError("No formally selected candidate is present in reference comparison")

    def box_alignment(velocities):
        xy = np.array(centers)[:, None, :] + contexts[None, :, None] * velocities[:, None, :]
        boxes = np.array(context_boxes)
        inside = ((boxes[..., :2] <= xy) & (xy <= boxes[..., 2:])).all(axis=-1)
        return {"context_centers_inside_box": int(inside.sum()), "context_observations": int(inside.size),
                "windows_all_16_inside": int(inside.all(axis=1).sum()), "eligible_windows": len(slopes)}

    return {"status": "PASS", "all_training_centers": total, "fitted_windows": len(slopes),
            "reference": "Pillow, SciPy 8-connected components, NumPy least squares and enumerated grid distances.",
            "candidates": counts, "continuous_fit_against_annotations": box_alignment(slopes),
            "selected_grid_against_annotations": box_alignment(chosen),
            "box_diagnostic": "Post-selection. Test each predicted centroid location against the corresponding "
                              "observed component's bounding box expanded to outer pixel edges (inclusive). "
                              "Ties in nearest velocity choose the lower component. A box hit is a permissive "
                              "geometric diagnostic; it does not establish a mask hit, identity or detection."}
