"""Retrieve only the preselected PNG members of the authors' pinned REDS training ZIP."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import io
import json
import struct
import time
import zipfile
import zlib


HERE = Path(__file__).resolve().parent
REVISION = "62dc25d16e6f43d2214f1b365023abda86f7a0ae"
URL = f"https://huggingface.co/datasets/snah/REDS/resolve/{REVISION}/train_sharp.zip"
ARCHIVE_BYTES = 34261573976
SEQUENCES = tuple(range(1, 240, 30))
FRAMES = range(48, 53)


def fetch(start, end):
    if not (0 <= start < end <= ARCHIVE_BYTES) or end - start > 32 * 2**20:
        raise ValueError(f"Unexpected archive range {start}:{end}")
    query = f"bytes={start}-{end-1}"
    req = Request(f"{URL}?range={start}-{end-1}", headers={"Range": query})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as response:
                expected = f"bytes {start}-{end-1}/{ARCHIVE_BYTES}"
                if response.status != 206 or response.headers["Content-Range"] != expected:
                    raise ValueError(f"Range response mismatch for {query}")
                data = response.read(end-start+1)
                if len(data) != end-start:
                    raise OSError(f"Short archive read for {query}")
                return data
        except (OSError, URLError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


class ArchiveIndexReader:
    """Read the ZIP64 central directory through zipfile with a cached archive tail."""
    def __init__(self):
        self.base = ARCHIVE_BYTES - 4 * 2**20
        self.tail = fetch(self.base, ARCHIVE_BYTES)
        self.pos = 0

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, ARCHIVE_BYTES)[whence] + offset
        return self.pos

    def read(self, count=-1):
        end = ARCHIVE_BYTES if count < 0 else min(self.pos + count, ARCHIVE_BYTES)
        if self.pos >= self.base:
            data = self.tail[self.pos-self.base:end-self.base]
        else:
            data = fetch(self.pos, end)
        self.pos = end
        return data

    def seekable(self):
        return True


def extract_member(args):
    info, end, seq, frame = args
    with io.BytesIO(fetch(info.header_offset, end)) as source:
        header = source.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Missing local header for {info.filename}")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        if source.read(name_len).decode("utf-8") != info.filename:
            raise ValueError(f"Local and central member names differ for {info.filename}")
        source.seek(extra_len, 1)
        payload = source.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -zlib.MAX_WBITS)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Unexpected compression method {info.compress_type}")
    if (len(payload) != info.file_size or zlib.crc32(payload) != info.CRC
            or payload[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"):
        raise ValueError(f"Archive CRC, size, or PNG signature differs for {info.filename}")
    width, height, depth, color = struct.unpack_from(">IIBB", payload, 16)
    if (width,height,depth,color) != (1280,720,8,2):
        raise ValueError(f"Unexpected PNG geometry for {info.filename}: {(width,height,depth,color)}")
    filename = Path("train_sharp") / f"{seq:03d}" / f"{frame:08d}.png"
    output = HERE / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(payload).hexdigest()
    if output.exists():
        if sha256(output.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing local member differs: {output}")
    else:
        temporary = output.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(output)
    print(filename, flush=True)
    return dict(path=str(filename), member=info.filename, size=info.file_size,
                crc32=f"{info.CRC:08x}", sha256=digest, sequence=seq, frame=frame,
                width=width, height=height, png_bit_depth=depth, png_color_type=color)


def main():
    with zipfile.ZipFile(ArchiveIndexReader()) as archive:
        sorted_info = sorted(archive.infolist(), key=lambda i: i.header_offset)
        ends = {a.filename:b.header_offset for a,b in zip(sorted_info, sorted_info[1:])}
        ends[sorted_info[-1].filename] = archive.start_dir
        known = {i.filename:i for i in sorted_info}
        items = []
        for seq in SEQUENCES:
            for frame in FRAMES:
                tail = f"/{seq:03d}/{frame:08d}.png"
                name, = [n for n in known if n.endswith(tail)]
                items.append((known[name], ends[name], seq, frame))
    with ThreadPoolExecutor(max_workers=8) as workers:
        records = list(workers.map(extract_member, items))
    result = dict(dataset="REDS", subset="train_sharp",
                  source="https://seungjunnah.github.io/Datasets/reds.html",
                  authors="Seungjun Nah et al., NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study",
                  license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                  archive_url=URL, revision=REVISION, archive_size=ARCHIVE_BYTES,
                  selection="Sequences 1+30k, k=0..7, frames 48 through 52 inclusive. Selected before viewing. Original unmodified RGB PNG members.",
                  files=records)
    manifest = HERE / "MANIFEST.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
