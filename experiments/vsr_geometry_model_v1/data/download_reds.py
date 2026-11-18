"""Extract the fixed two-way REDS split from the original authors' pinned archive."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib


HERE = Path(__file__).resolve().parent
OLD_INPUT = HERE.parents[1] / "vsr_sampling_x2_20261118" / "inputs"
FETCH_SCRIPT = OLD_INPUT / "download_reds.py"
SELECTION = {
    "train": ((2,12,22,32,42,52,62,72,82,92,102,112), (32,72)),
    "evaluation": ((9,39,69,99,129,159,189,219), (50,)),
}
spec = importlib.util.spec_from_file_location("reds_zip_ranges", FETCH_SCRIPT)
source = importlib.util.module_from_spec(spec)
save_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    spec.loader.exec_module(source)
finally:
    sys.dont_write_bytecode = save_bytecode
source.HERE = HERE


def verify_png(payload):
    """Validate file integrity without inspecting content or computing scene statistics."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Missing PNG signature")
    pos, raw_bytes = 8, 0
    decoder = zlib.decompressobj()
    while pos < len(payload):
        size, = struct.unpack_from(">I", payload, pos)
        kind = payload[pos+4:pos+8]
        block = payload[pos+8:pos+8+size]
        checksum, = struct.unpack_from(">I", payload, pos+8+size)
        if zlib.crc32(kind + block) != checksum:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IDAT":
            raw_bytes += len(decoder.decompress(block))
        pos += 12 + size
        if kind == b"IEND":
            break
    raw_bytes += len(decoder.flush())
    geometry = struct.unpack_from(">IIBBBBB", payload, 16)
    if (geometry != (1280,720,8,2,0,0,0) or not decoder.eof or decoder.unused_data
            or kind != b"IEND" or pos != len(payload) or raw_bytes != 720 * (1+3*1280)):
        raise ValueError("Unexpected PNG layout or incomplete decoded stream")


def obtain(args):
    info, end, split, seq, center, frame = args
    name = Path("train_sharp") / f"{seq:03d}" / f"{frame:08d}.png"
    path = HERE / name
    if not path.exists():
        source.extract_member((info,end,seq,frame))
    data = path.read_bytes()
    if (len(data),zlib.crc32(data)) != (info.file_size,info.CRC):
        raise ValueError(f"Local file differs from source archive: {path}")
    verify_png(data)
    return dict(path=str(name), member=info.filename, sequence=seq, frame=frame, center=center,
                split=split, size=info.file_size, crc32=f"{info.CRC:08x}", sha256=sha256(data).hexdigest(),
                width=1280,height=720,png_bit_depth=8,png_color_type=2)


def main():
    pilot = json.loads((OLD_INPUT/"MANIFEST.json").read_text())
    train, evaluation = set(SELECTION["train"][0]), set(SELECTION["evaluation"][0])
    if train & evaluation or (train | evaluation) & {r["sequence"] for r in pilot["files"]}:
        raise ValueError("Splits overlap")
    with zipfile.ZipFile(source.ArchiveIndexReader()) as archive:
        infos = sorted(archive.infolist(),key=lambda i:i.header_offset)
        ends = {i.filename:j.header_offset for i,j in zip(infos, infos[1:])}
        ends[infos[-1].filename] = archive.start_dir
        items = []
        for split,(sequences,centers) in SELECTION.items():
            for seq in sequences:
                for center in centers:
                    for frame in range(center-2,center+3):
                        name=f"train/train_sharp/{seq:03d}/{frame:08d}.png"
                        items.append((archive.getinfo(name),ends[name],split,seq,center,frame))
    with ThreadPoolExecutor(max_workers=8) as pool:
        files=list(pool.map(obtain,items))
    manifest = {k:pilot[k] for k in ("dataset","subset","source","authors","license","license_url",
                                      "archive_url","revision","archive_size")}
    manifest.update(selection={s:dict(sequences=seq, centers=c,frame_offsets=[-2,-1,0,1,2])
                               for s,(seq,c) in SELECTION.items()}, files=files,
                    source_reader_sha256=sha256(FETCH_SCRIPT.read_bytes()).hexdigest(),
                    verification="Member size and ZIP CRC; all internal PNG chunk CRCs; complete decoded byte count; per-file SHA-256")
    target = HERE/"MANIFEST.json"
    pending=target.with_suffix(".tmp")
    pending.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    pending.replace(target)
    print(f"Verified {len(files)} files, {sum(f['size'] for f in files)} bytes",flush=True)


if __name__ == "__main__":
    main()
