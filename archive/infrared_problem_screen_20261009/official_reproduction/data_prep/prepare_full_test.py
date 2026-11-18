"""Acquire only the official NUDT-MIRSDT test PNG triples via public HTTP Range.

Split sequence order and frame names come from the pinned upstream test.txt.
Archive central-directory CRC32 values verify every extracted file, including ZIP
members whose local headers use data descriptors. No training data is extracted.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "official_reproduction"
PREP = OUT / "data_prep"
DEST = OUT / "data"
DATA = DEST / "NUDT-MIRSDT"
SOURCE = ROOT / "official_deeppro"
BUDGET = 400_000_000
KINDS = ("images", "masks", "masks_centroid")
LOW_SNR = ["Sequence" + str(n) for n in (47, 56, 59, 76, 92, 101, 105, 119)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_central(blob):
    pos = 0
    entries = {}
    while pos + 46 <= len(blob):
        h = struct.unpack_from("<4s6H3L5H2L", blob, pos)
        assert h[0] == b"PK\x01\x02"
        flags, method, crc, compressed, size = h[3], h[4], h[7], h[8], h[9]
        nl, el, cl = h[10:13]
        name = blob[pos + 46:pos + 46 + nl].decode("utf-8" if flags & 0x800 else "cp437")
        entries[name] = dict(name=name, flags=flags, method=method, crc32=f"{crc:08x}",
                             compressed_bytes=compressed, uncompressed_bytes=size,
                             local_header_offset=h[16])
        pos += 46 + nl + el + cl
    assert pos == len(blob)
    return entries


def main():
    PREP.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    split_bytes = (SOURCE / "datasets/NUDT-MIRSDT/test.txt").read_bytes()
    lines = split_bytes.decode().splitlines()
    assert len(lines) == len(set(lines))
    sequence_order = list(dict.fromkeys(line.split("/")[0] for line in lines))
    expected = {seq: [PurePosixPath(line).stem + ".png" for line in lines
                      if line.split("/")[0] == seq] for seq in sequence_order}
    assert all(names == sorted(names) and len(names) == len(set(names)) for names in expected.values())
    assert set(LOW_SNR).issubset(sequence_order)
    (DATA / "test.txt").write_bytes(split_bytes)
    manifest_path = ROOT / "NUDT_PUBLIC_ZIP_MANIFEST.json"
    original = json.loads(manifest_path.read_text())
    selected_names = {f"NUDT-MIRSDT/{seq}/{kind}/{name}" for seq in sequence_order
                      for kind in KINDS for name in expected[seq]}
    original_entries = {e["name"]: e for e in original["entries"]}
    assert selected_names.issubset(original_entries)
    helper = ROOT / "runtime/fetch_public_zip_subset.py"
    spec = importlib.util.spec_from_file_location("public_zip_helper", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    state_path = PREP / "DOWNLOAD_STATE.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_body_bytes_received": 0, "public_form_budget_reserve_bytes": 0,
        "completed_groups": [], "status": "running"}
    prior_bytes = state["range_body_bytes_received"]
    state["public_form_budget_reserve_bytes"] += 20_000
    arc = mod.PublicArchive(original, BUDGET - prior_bytes - state["public_form_budget_reserve_bytes"])

    def save_state():
        state["range_body_bytes_received"] = prior_bytes + arc.transferred
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    save_state()
    central_path = PREP / "ARCHIVE_CENTRAL_DIRECTORY.bin"
    if not central_path.exists():
        archive_size = original["archive_bytes"]
        tail = arc.read_range(max(0, archive_size - 65557), archive_size - 1)
        save_state()
        pos = tail.rfind(b"PK\x05\x06")
        assert pos >= 0
        eocd = struct.unpack_from("<4s4H2LH", tail, pos)
        assert eocd[1] == eocd[2] == 0 and eocd[3] == eocd[4]
        count, central_size, central_offset = eocd[4:7]
        assert count == original["entries_count"] and central_size == original["central_directory_bytes"]
        central_blob = arc.read_range(central_offset, central_offset + central_size - 1)
        save_state()
        central_path.write_bytes(central_blob)
        write_json(PREP / "CENTRAL_DIRECTORY_PROVENANCE.json", {
            "source_url": arc.source_url, "http_status": 206,
            "central_directory_offset": central_offset, "central_directory_bytes": central_size,
            "sha256": sha(central_blob), "footer_range_bytes": len(tail), "entries_count": count})
    central_blob = central_path.read_bytes()
    central_prov = json.loads((PREP / "CENTRAL_DIRECTORY_PROVENANCE.json").read_text())
    assert sha(central_blob) == central_prov["sha256"]
    central = parse_central(central_blob)
    selected = sorted((central[name] for name in selected_names), key=lambda e: e["local_header_offset"])
    for e in selected:
        for key, value in original_entries[e["name"]].items():
            assert e[key] == value, (e["name"], key)
    write_json(PREP / "SELECTED_ARCHIVE_MEMBERS.json", selected)
    missing = []
    for e in selected:
        path = DEST.joinpath(*mod.safe_path(e["name"]).parts)
        if path.exists():
            data = path.read_bytes()
            assert len(data) == e["uncompressed_bytes"] and f"{zlib.crc32(data) & 0xffffffff:08x}" == e["crc32"]
        else:
            missing.append(e)
    groups = []
    for e in missing:
        start = e["local_header_offset"]
        end = min(original["archive_bytes"] - 1, start + 30 + len(e["name"].encode()) + 65535 + e["compressed_bytes"])
        if groups and start - groups[-1][1] < 65536 and end - groups[-1][0] <= 8_000_000:
            groups[-1][1] = max(end, groups[-1][1])
            groups[-1][2].append(e)
        else:
            groups.append([start, end, [e]])
    estimate = sum(end - start + 1 for start, end, _ in groups)
    assert estimate + state["range_body_bytes_received"] + state["public_form_budget_reserve_bytes"] <= BUDGET
    print(json.dumps({"split_sequences": len(sequence_order), "split_frames": len(lines),
                      "required_png_files": len(selected), "missing_png_files": len(missing),
                      "planned_range_bytes": estimate, "groups": len(groups)}), flush=True)
    for group_idx, (start, end, items) in enumerate(groups):
        blob = arc.read_range(start, end)
        save_state()
        for e in items:
            data = arc.member_from_blob(e, blob, start)
            crc = f"{zlib.crc32(data) & 0xffffffff:08x}"
            assert crc == e["crc32"], (e["name"], "central-directory CRC mismatch")
            path = DEST.joinpath(*mod.safe_path(e["name"]).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        state["completed_groups"].append({"first": items[0]["name"], "last": items[-1]["name"],
                                          "members": len(items), "start": start, "end": end})
        save_state()
        print(json.dumps({"group": group_idx + 1, "groups": len(groups),
                          "members": len(items), "range_body_bytes_received": state["range_body_bytes_received"]}), flush=True)
    records = []
    dimensions = collections.Counter()
    modes = collections.Counter()
    per_sequence = []
    for seq in sequence_order:
        names = expected[seq]
        for kind in KINDS:
            actual = sorted(p.name for p in (DATA / seq / kind).iterdir())
            assert actual == names, (seq, kind, "unexpected names or order")
        shapes = collections.Counter()
        for name in names:
            frame_shapes = []
            for kind in KINDS:
                member = f"NUDT-MIRSDT/{seq}/{kind}/{name}"
                e = central[member]
                data = (DEST / member).read_bytes()
                crc = f"{zlib.crc32(data) & 0xffffffff:08x}"
                assert len(data) == e["uncompressed_bytes"] and crc == e["crc32"]
                with Image.open(io.BytesIO(data)) as img:
                    size, mode = img.size, img.mode
                    img.verify()
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
                frame_shapes.append(size)
                dimensions[str(size)] += 1
                modes[mode] += 1
                records.append({"member": member, "bytes": len(data), "sha256": sha(data),
                                "crc32": crc, "archive_central_crc32": e["crc32"],
                                "crc_verified": True, "size_wh": size, "mode": mode,
                                "png_verified_and_decoded": True})
            assert len(set(frame_shapes)) == 1, (seq, name, "triple shape mismatch")
            shapes[str(frame_shapes[0])] += 1
        per_sequence.append({"sequence": seq, "frames": len(names), "files": len(names) * 3,
                             "first": names[0], "last": names[-1], "dimensions_wh": dict(shapes),
                             "snr_le_3_official_subset": seq in LOW_SNR,
                             "pairings_complete": True, "lexical_order_matches_split_frame_order": True})
    train_file = SOURCE / "datasets/NUDT-MIRSDT/train.txt"
    train_sequences = set(line.split("/")[0] for line in train_file.read_text().splitlines())
    assert not train_sequences.intersection(sequence_order)
    actual_sequences = {p.name for p in DATA.iterdir() if p.is_dir()}
    assert actual_sequences == set(sequence_order)
    assert (DATA / "test.txt").read_bytes() == split_bytes
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    write_json(PREP / "FILE_CHECKSUMS.json", records)
    state["status"] = "complete"
    save_state()
    audit = {
        "status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATA), "official_repository": "https://github.com/TinaLRJ/DeepPro",
        "official_commit": commit, "split_source": "datasets/NUDT-MIRSDT/test.txt",
        "split_sha256": sha(split_bytes), "split_preserved_byte_for_byte": True,
        "split_lines": len(lines), "unique_sequences": len(sequence_order),
        "sequence_order": sequence_order, "frame_names_mapped_from_official_mat_stems": True,
        "official_loader_uses_sequence_names_then_sorted_png_files": True,
        "png_files": len(records), "counts_by_kind": {kind: len(lines) for kind in KINDS},
        "bytes_extracted": sum(r["bytes"] for r in records),
        "dimensions_wh": dict(dimensions), "modes": dict(modes),
        "all_triples_paired": True, "all_dimensions_match_within_triples": True,
        "all_pngs_verified_and_decoded": True, "all_archive_crcs_verified": True,
        "all_sha256_recorded": True, "unexpected_files_in_frame_folders": [],
        "missing_official_test_frames": [], "train_test_sequence_intersection": [],
        "training_sequences_extracted": [], "snr_le_3_sequence_list": LOW_SNR,
        "snr_le_3_sequences": len(LOW_SNR),
        "snr_le_3_frames": sum(len(expected[seq]) for seq in LOW_SNR),
        "snr_subset_source": f"https://github.com/TinaLRJ/DeepPro/blob/{commit}/README.md#datasets",
        "snr_values_not_reestimated": True, "per_sequence": per_sequence,
        "provenance": {"source_url": arc.source_url, "official_public_form_used": True,
                       "range_only": True, "http_206_and_content_range_verified": True,
                       "archive_bytes": original["archive_bytes"],
                       "existing_archive_manifest_sha256": sha(manifest_path.read_bytes()),
                       "central_directory_sha256": sha(central_blob),
                       "central_directory_crc_authority": True,
                       "range_body_bytes_received": state["range_body_bytes_received"],
                       "public_form_upper_bound_bytes": state["public_form_budget_reserve_bytes"],
                       "total_network_upper_bound_bytes": state["range_body_bytes_received"] + state["public_form_budget_reserve_bytes"],
                       "network_budget_bytes": BUDGET,
                       "member_checksum_manifest": str(PREP / "FILE_CHECKSUMS.json")}}
    write_json(OUT / "FULL_TEST_DATA_AUDIT.json", audit)
    rows = "\n".join(f"| {s['sequence']} | {s['frames']} | {s['files']} | {'yes' if s['snr_le_3_official_subset'] else 'no'} |"
                     for s in per_sequence)
    md = f"""# Full official NUDT-MIRSDT test data audit

**PASS** — all {len(lines):,} frames from all {len(sequence_order)} official test sequences are present as {len(records):,} original PNG files.

- Data root: `{DATA}`
- Official repository: https://github.com/TinaLRJ/DeepPro, pinned commit `{commit}`.
- Authoritative split: `datasets/NUDT-MIRSDT/test.txt`; copied byte for byte, SHA-256 `{sha(split_bytes)}`.
- Official archive: {arc.source_url}; archive size {original['archive_bytes']:,} bytes.
- Range bodies received: {state['range_body_bytes_received']:,} bytes; total body upper bound including public download form: {audit['provenance']['total_network_upper_bound_bytes']:,} bytes, within the {BUDGET:,}-byte budget.
- Only official test PNG triples were extracted; no training sequence was extracted. Public ZIP range requests can include small adjacent bytes to cover variable ZIP headers.
- Images, masks and centroid masks have matching names and dimensions for every frame. Each PNG passed Pillow verification and full decoding.
- All 6,000 CRC32 values match the archive's central directory; SHA-256, dimensions and modes are recorded per file in `data_prep/FILE_CHECKSUMS.json`.
- Dimensions (width, height), counted across all three kinds: `{dict(dimensions)}`; modes: `{dict(modes)}`.
- The `.mat` entries in the official split identify frame stems. The official loader takes sequence names in first occurrence order, then sorts the PNG filenames. The downloaded layout matches both conventions.
- Official SNR≤3 subset: {', '.join(LOW_SNR)} ({audit['snr_le_3_frames']} frames). Membership comes from the pinned official README, without reestimating SNR.

## Official sequence order and coverage

| Sequence | Frames | PNG files | SNR≤3 |
|---|---:|---:|---|
{rows}

## Reproducibility artifacts

- `data_prep/prepare_full_test.py`: bounded acquisition and verification script.
- `data_prep/SELECTED_ARCHIVE_MEMBERS.json`: exact ZIP members and central CRC32 values.
- `data_prep/CENTRAL_DIRECTORY_PROVENANCE.json` and `ARCHIVE_CENTRAL_DIRECTORY.bin`: CRC authority and metadata hash.
- `data_prep/DOWNLOAD_STATE.json`: completed groups and transferred range bytes.
- `data_prep/FILE_CHECKSUMS.json`: per-file SHA-256, CRC32, sizes and decode results.

Archive authenticity is tied to the URL linked by the pinned official repository. Per-file SHA-256 values establish local reproducibility; no upstream cryptographic digest is published or asserted.
"""
    (OUT / "FULL_TEST_DATA_AUDIT.md").write_text(md)
    print(json.dumps({"status": "PASS", "frames": len(lines), "png_files": len(records),
                      "audit": str(OUT / "FULL_TEST_DATA_AUDIT.json")}), flush=True)


if __name__ == "__main__":
    main()
