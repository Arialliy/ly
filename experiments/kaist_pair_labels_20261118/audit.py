"""Audit the AR-CNN annotation ZIP in place. Run with python3 audit.py."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ZIP = HERE / "annotations.zip"
REV = "dd1d8d56d4fffeba78c380aa9abde17b92ed2021"
REPO = "https://github.com/luzhang16/AR-CNN"
RAW = f"https://raw.githubusercontent.com/luzhang16/AR-CNN/{REV}"
PAPER = "https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html"
ENTRY = re.compile(r"kaist-paired/annotations/(set\d+/V\d+)/(lwir|visible)/(I\d+)\.txt")
GROUPS = ("paired_person", "lwir_unpaired", "visible_unpaired")
SHA = "615a747af66d03a3dcc13a139bf868248dbaf6c1b6b42fb64600dbf5b9e921fb"


def get_splits():
    membership, sources = {}, {}
    for split in ("trainval", "test"):
        url = f"{RAW}/splits/{split}.txt"
        with urlopen(url, timeout=30) as response:
            content = response.read()
        frames = content.decode("ascii").splitlines()
        assert len(frames) == len(set(frames))
        assert set(frames).isdisjoint(membership)
        membership.update(dict.fromkeys(frames, split))
        sources[split] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "frames": len(frames)}
    return membership, sources


def read_rows(archive, name, classes, occ):
    header, *lines = archive.read(name).decode("ascii").splitlines()
    assert header == "% bbGt version=3"
    rows = [line.split() for line in lines if line.strip()]
    for row in rows:
        assert len(row) == 12 and row[6:] == ["0"] * 6
        x, y, w, h, flag = map(int, row[1:6])
        assert w > 0 and h > 0 and flag in (0, 1, 2)
        classes[row[0]] += 1
        occ[flag] += 1
    return rows


def report(by_sequence):
    counts = sum(by_sequence.values(), Counter())
    return {
        "frames": counts["frames"], "sequences": len(by_sequence),
        **{group: {"rows": counts[group], "frames": counts[group + "_frames"],
                   "sequences": sum(s[group] > 0 for s in by_sequence.values())}
           for group in GROUPS},
        "paired_person_identical_xywh": counts["identical_xywh"],
        "frames_with_any_lwir_visible_occlusion_flag_difference_after_removing_unpaired": counts["different_flags"],
        "by_sequence": {key: dict(counts) for key, counts in sorted(by_sequence.items())},
    }


def main():
    assert ZIP.stat().st_size == 28449339
    assert hashlib.sha256(ZIP.read_bytes()).hexdigest() == SHA
    membership, splits = get_splits()
    class_counts, flags = defaultdict(Counter), defaultdict(Counter)
    grouped = defaultdict(lambda: defaultdict(Counter))
    with ZipFile(ZIP) as archive:
        files = defaultdict(dict)
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            seq, mod, frame = ENTRY.fullmatch(entry.filename).groups()
            assert mod not in files[seq, frame]
            files[seq, frame][mod] = entry.filename
        assert set(membership) <= {f"{s}/{f}" for s, f in files}
        for (seq, frame), names in files.items():
            assert set(names) == {"lwir", "visible"}
            metrics = Counter(frames=1)
            paired = {}
            for mod, name in names.items():
                rows = read_rows(archive, name, class_counts[mod], flags[mod])
                paired[mod] = [row for row in rows if row[0] != "unpaired"]
                metrics[mod + "_unpaired"] = len(rows) - len(paired[mod])
            thermal, visible = paired["lwir"], paired["visible"]
            assert [r[0] for r in thermal] == [r[0] for r in visible]
            metrics["different_flags"] = int(any(a[5] != b[5] for a, b in zip(thermal, visible)))
            people = [(a, b) for a, b in zip(thermal, visible) if a[0] == "person"]
            metrics["paired_person"] = len(people)
            metrics["identical_xywh"] = sum(a[1:5] == b[1:5] for a, b in people)
            for group in GROUPS:
                metrics[group + "_frames"] = int(metrics[group] > 0)
            split = membership.get(f"{seq}/{frame}", "outside_author_lists")
            grouped["archive"][seq].update(metrics)
            grouped[split][seq].update(metrics)
    output = {
        "sources": {"readme": f"{REPO}/blob/{REV}/README.md", "paper": PAPER,
                    "parser": f"{REPO}/blob/{REV}/lib/datasets/kaist.py#L208-L284",
                    "zip": "https://drive.google.com/open?id=1FLkoJQOGt4PqRtr0j6namAaehrdt_A45",
                    "bytes": ZIP.stat().st_size, "sha256": SHA, "official_splits": splits},
        "archive_class_rows_by_modality": class_counts,
        "archive_column_6_counts_by_modality": flags,
        "field_checks": "Every frame has a file in each modality. Every row has class, integer xywh, flag 0/1/2 and six zero fields. After dropping unpaired, row counts and class order agree for all frame pairs.",
        "counting": "One unpaired row in a named modality file counts as one object-frame annotated only in that modality under paper section 3.3. Paired person counts take literal person rows once following the shared order. A sequence is a set/video; frames may repeat people. Archive totals include frames outside the authors' trainval/test lists.",
        "limits": [
            "The released loader discards unpaired. Its default filter takes person rows with flag other than 2 and relabels cyclists as person. The audit counts literal classes before that filter.",
            "The annotation policy explicitly copies thermal coordinates into color boxes under extreme darkness. No per-box copy indicator is populated; matching coordinates alone cannot identify those events.",
            "Single-modality labels can arise from field-of-view differences. The archive contains no teacher predictions and supplies no teacher-miss labels or verified sensor-visibility labels for paired boxes.",
            "Counts are object-frame annotations; the observed fields contain no nonconstant identifier for tracking across frames.",
        ],
        "coverage": {name: report(data) for name, data in grouped.items()},
    }
    (HERE / "audit.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
