"""Annotation-only segment grouping. No model scores or test labels are used."""
from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence


@dataclass(frozen=True)
class Observation:
    video: str
    track: str
    frame: int
    row: int
    visible_and_labeled: bool = True


def visible_segments(
    observations: Sequence[Observation], *, length: int, step: int,
    stride: int | None = None,
) -> list[tuple[int, ...]]:
    """Return row-index tuples within a single video/track and exact frame cadence.

    Invisibility, unlabelled observations and missing frame IDs break segments.
    Only full-length segments are returned; excluded rows still retain ordinary
    detection supervision in the caller. Fix length/step/stride on training data
    before inspecting development performance.
    """
    stride = length if stride is None else stride
    if length < 2 or step < 1 or stride < 1:
        raise ValueError('length>=2, step>=1, stride>=1 required')
    grouped = defaultdict(list)
    seen, rows = set(), set()
    for o in observations:
        key = (o.video, o.track, o.frame)
        if key in seen or o.row in rows:
            raise ValueError('duplicate observation key or row')
        seen.add(key)
        rows.add(o.row)
        grouped[(o.video, o.track)].append(o)

    result: list[tuple[int, ...]] = []
    def flush(run):
        for i in range(0, len(run) - length + 1, stride):
            result.append(tuple(x.row for x in run[i:i + length]))

    for key in sorted(grouped):
        run = []
        for o in sorted(grouped[key], key=lambda x: x.frame):
            if not o.visible_and_labeled:
                flush(run)
                run = []
                continue
            if run and o.frame - run[-1].frame != step:
                flush(run)
                run = []
            run.append(o)
        flush(run)
    return result
