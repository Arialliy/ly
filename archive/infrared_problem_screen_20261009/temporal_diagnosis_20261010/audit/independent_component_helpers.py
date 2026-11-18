"""Functions from this audit's independent recheck, not original analysis code."""
import numpy as np

def flood_components(p, th=0.5):
    h, w = p.shape
    active = np.flatnonzero(p.ravel() >= th).tolist()
    unseen = set(active)
    components = []
    for origin in active:
        if origin not in unseen:
            continue
        unseen.remove(origin)
        stack = [origin]
        pixels = []
        while stack:
            idx = stack.pop()
            pixels.append(idx)
            y, x = divmod(idx, w)
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    other = yy * w + xx
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        ids = np.asarray(sorted(pixels), dtype=np.int64)
        values = p.ravel()[ids]
        best = float(values.max())
        peaks = ids[values == best]
        py, px = divmod(int(peaks[0]), w)
        components.append({'id': len(components) + 1, 'x': px + 0.5, 'y': py + 0.5, 'score': best, 'area': len(pixels), 'peak_ties': len(peaks), 'peak_indices': peaks})
    return components

def assignment(gts, comps):
    edges = []
    for a in gts:
        x, y, w, h = a['bbox']
        cx, cy = (x + w / 2, y + h / 2)
        choices = [i for i, c in enumerate(comps) if x <= c['x'] < x + w and y <= c['y'] < y + h]
        choices.sort(key=lambda i: (((comps[i]['x'] - cx) / w) ** 2 + ((comps[i]['y'] - cy) / h) ** 2, -comps[i]['score'], comps[i]['id']))
        edges.append(choices)
    owner = {}

    def augment(gi, seen):
        for ci in edges[gi]:
            if ci in seen:
                continue
            seen.add(ci)
            if ci not in owner or augment(owner[ci], seen):
                owner[ci] = gi
                return True
        return False
    for gi in range(len(gts)):
        augment(gi, set())
    masks = {0}
    for ci in range(len(comps)):
        candidates = [g for g, e in enumerate(edges) if ci in e]
        if candidates:
            expanded = set(masks)
            for old in masks:
                for g in candidates:
                    if not old >> g & 1:
                        expanded.add(old | 1 << g)
            masks = expanded
    max_count = max((int(m).bit_count() for m in masks))
    assert max_count == len(owner)
    return ({g: c for c, g in owner.items()}, edges)
