#!/usr/bin/env python3
"""Generate or independently verify a complete, quiescent package manifest.

Only PACKAGE_MANIFEST.json is excluded. No data/model code is imported. Invoke
after every writer has finished; stdout must not be redirected into this tree
while generating/verifying. All generated backup/temp files stay under --root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


NAME = "PACKAGE_MANIFEST.json"
SCHEMA = "complete_package_manifest_v1"
STAT_FIELDS = ("device", "inode", "size", "mtime_ns", "ctime_ns")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def signature(s):
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def key(s):
    return tuple(s[k] for k in STAT_FIELDS)


def scan(root):
    """Never traverse symlinks; enumerate everything except the manifest itself."""
    files, links, directories = {}, {}, []
    pending = [root]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as iterator:
            entries = sorted(iterator, key=lambda e: e.name)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if relative == NAME:
                continue
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError(f"External symlink: {relative} -> {resolved}")
                links[relative] = dict(target=target,
                                       resolved_relative=resolved.relative_to(root).as_posix(),
                                       valid=True, internal=True)
            elif stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                files[relative] = signature(info)
            else:
                raise ValueError(f"Unsupported non-regular entry: {relative}")
    return dict(files=files, symlinks=links, directories=sorted(directories))


def hash_regular(path, expected):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        if signature(os.fstat(stream.fileno())) != expected:
            raise RuntimeError(f"File changed before hashing: {path}")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if signature(os.fstat(stream.fileno())) != expected:
            raise RuntimeError(f"File changed while hashing: {path}")
    if signature(path.stat(follow_symlinks=False)) != expected:
        raise RuntimeError(f"File replaced while hashing: {path}")
    return digest.hexdigest()


def previous_and_backup(root):
    output = root / NAME
    if not output.exists() and not output.is_symlink():
        return None, None
    if output.is_symlink() or not output.is_file():
        raise ValueError("Existing PACKAGE_MANIFEST.json must be a regular file")
    raw = output.read_bytes()
    prior_hash = hashlib.sha256(raw).hexdigest()
    backup_dir = root / "provenance"
    backup_dir.mkdir(exist_ok=True)
    if backup_dir.is_symlink():
        raise ValueError("provenance directory must not be a symlink")
    name = f"PACKAGE_MANIFEST_before_{time.time_ns()}.json"
    backup = backup_dir / name
    with backup.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Preserve an invalid or pending old manifest too, but never use it as cache.
    try:
        previous = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        previous = None
    return previous, dict(path=backup.relative_to(root).as_posix(), sha256=prior_hash,
                          bytes=len(raw), byte_preserving_backup=True)


def prior_cache(previous):
    cache = {}
    if not isinstance(previous, dict) or previous.get("schema") != SCHEMA:
        return cache
    if previous.get("status") != "COMPLETE":
        return cache
    for row in previous.get("files", []):
        digest = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Prior manifest contains invalid SHA256")
        entry_key = key(row["stat"])
        if entry_key in cache and cache[entry_key] != digest:
            raise ValueError("Prior manifest contradicts itself for an inode")
        cache[entry_key] = digest
    return cache


def totals(snapshot):
    unique = {}
    for row in snapshot["files"].values():
        inode = (row["device"], row["inode"])
        if inode in unique and unique[inode] != row["size"]:
            raise RuntimeError("Conflicting hardlink sizes; tree is not quiescent")
        unique[inode] = row["size"]
    return dict(regular_file_count=len(snapshot["files"]),
                symlink_count=len(snapshot["symlinks"]),
                directory_count=len(snapshot["directories"]),
                logical_regular_file_bytes=sum(r["size"] for r in snapshot["files"].values()),
                unique_regular_inode_count=len(unique),
                unique_regular_inode_bytes=sum(unique.values()),
                symlink_target_text_bytes=sum(len(os.fsencode(r["target"]))
                                               for r in snapshot["symlinks"].values()),
                byte_scope="Regular-file content only; hardlinks counted once in unique bytes. Symlink targets are not traversed or double-counted.")


def assert_unchanged(root, before):
    after = scan(root)
    if after != before:
        changed = []
        for category in ["files", "symlinks"]:
            a, b = before[category], after[category]
            changed += [f"{category}:{p}" for p in set(a) | set(b) if a.get(p) != b.get(p)]
        changed += ["directory:" + p for p in
                    set(before["directories"]) ^ set(after["directories"])]
        raise RuntimeError("Package changed during operation; no final manifest written: "
                           + ", ".join(sorted(changed)[:20]))


def atomic_manifest(root, value):
    path = root / NAME
    temporary = root / (".PACKAGE_MANIFEST.write_" + str(time.time_ns()) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate(root, reuse):
    previous, backup = previous_and_backup(root)
    cache = prior_cache(previous) if reuse else {}
    before = scan(root)
    computed, previous_reused = {}, set()
    rows = []
    for relative, info in sorted(before["files"].items()):
        entry_key = key(info)
        if entry_key in computed:
            digest = computed[entry_key]
        elif entry_key in cache:
            digest = cache[entry_key]
            computed[entry_key] = digest
            previous_reused.add(entry_key)
        else:
            digest = hash_regular(root / relative, info)
            computed[entry_key] = digest
        rows.append(dict(path=relative, bytes=info["size"], sha256=digest, stat=info))
    assert_unchanged(root, before)
    value = dict(schema=SCHEMA, status="COMPLETE", created_utc=utc(),
                 root_at_generation=str(root), excluded_relative_paths=[NAME],
                 previous_manifest=backup,
                 hash_policy={
                     "algorithm": "SHA256",
                     "unique_inode_hashing": True,
                     "prior_cache_enabled": reuse,
                     "prior_reuse_basis": "Only this schema's COMPLETE prior manifest; identical device,inode,size,mtime_ns,ctime_ns. Prior manifest preserved byte-for-byte and SHA-bound in previous_manifest. No upstream collection hash is trusted solely by pathname/size.",
                     "reused_prior_unique_inodes": len(previous_reused),
                     "directly_hashed_unique_inodes": len(computed) - len(previous_reused),
                     "quiescence_check": "Full path/type/file-stat/symlink scan before and after hashing. Writers must already be stopped; filesystem snapshot/locking is not provided.",
                     "verification": "verify independently rehashes every unique inode by default; inode metadata is not required to match after copying the package."},
                 statistics=totals(before), files=rows,
                 symlinks=[dict(path=p, **r) for p, r in sorted(before["symlinks"].items())],
                 directories=before["directories"])
    atomic_manifest(root, value)
    return dict(status="COMPLETE", manifest=str(root / NAME),
                manifest_sha256=hashlib.sha256((root / NAME).read_bytes()).hexdigest(),
                statistics=value["statistics"], hash_policy=value["hash_policy"])


def verify(root):
    path = root / NAME
    if path.is_symlink():
        raise ValueError("Manifest must not be a symlink")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "COMPLETE":
        raise ValueError("A COMPLETE manifest from this tool is required")
    if manifest.get("excluded_relative_paths") != [NAME]:
        raise ValueError("Unexpected manifest exclusions")
    expected_files = {r["path"]: r for r in manifest["files"]}
    expected_links = {r["path"]: {k: v for k, v in r.items() if k != "path"}
                      for r in manifest["symlinks"]}
    if len(expected_files) != len(manifest["files"]) or len(expected_links) != len(manifest["symlinks"]):
        raise ValueError("Duplicate manifest paths")
    before = scan(root)
    if set(before["files"]) != set(expected_files):
        missing = set(expected_files) - set(before["files"])
        extra = set(before["files"]) - set(expected_files)
        raise ValueError(f"File set mismatch; missing={sorted(missing)[:20]}, extra={sorted(extra)[:20]}")
    if before["symlinks"] != expected_links or before["directories"] != manifest["directories"]:
        raise ValueError("Symlink/directory set or symlink target mismatch")
    hashes = {}
    for relative, info in sorted(before["files"].items()):
        expected = expected_files[relative]
        if info["size"] != expected["bytes"]:
            raise ValueError("Size mismatch: " + relative)
        entry_key = key(info)
        if entry_key not in hashes:
            hashes[entry_key] = hash_regular(root / relative, info)
        if hashes[entry_key] != expected["sha256"]:
            raise ValueError("SHA256 mismatch: " + relative)
    assert_unchanged(root, before)
    if path.read_bytes() != raw:
        raise RuntimeError("Manifest changed during verification")
    actual = totals(before)
    for name in ["regular_file_count", "symlink_count", "directory_count", "logical_regular_file_bytes", "symlink_target_text_bytes"]:
        if actual[name] != manifest["statistics"][name]:
            raise ValueError("Manifest statistics mismatch: " + name)
    return dict(status="PASS", manifest=str(path),
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                current_statistics=actual,
                recorded_statistics=manifest["statistics"],
                directly_hashed_unique_inodes=len(hashes),
                note="Content, paths and internal symlinks verified. Hardlink topology may differ after copying; recorded and current unique-inode bytes are reported separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-prior-stat-cache", action="store_true",
                        help="Generation only: reuse prior COMPLETE-manifest hashes with unchanged full inode stat key. Default rehashes each unique inode.")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        parser.error("--root must be an existing directory")
    if args.action == "verify" and args.reuse_prior_stat_cache:
        parser.error("verify always hashes unique inodes independently")
    try:
        result = generate(root, args.reuse_prior_stat_cache) if args.action == "generate" else verify(root)
    except Exception as error:
        print(json.dumps(dict(status="FAIL", action=args.action, root=str(root),
                              error=type(error).__name__ + ": " + str(error)), ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
