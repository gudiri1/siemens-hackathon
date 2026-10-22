"""Download models from the Vascular Model Repository (vascularmodel.com).

VMR serves one zip per model. The zips are large (100 MB to 17 GB), but the server supports HTTP
Range requests, so we read only the zip's central directory and then fetch and inflate the few
entries we need: the centerline paths (``Paths/*.pth``), the surface model (``Models/*.vtp``) and
the ``LICENSE`` file. Downloaded files are treated as data only and written under ``data/raw/``.
"""

from __future__ import annotations

import csv
import io
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "https://www.vascularmodel.com/"
INDEX_URL = BASE_URL + "dataset/dataset-svprojects.csv"
DEFAULT_RAW_DIR = Path("data/raw")

# Models whose download layout differs from the generic ``svprojects/{name}.zip``.
# 0248 ships a small "reference" zip with one variant and a 17 GB zip with all 30 variants.
SPECIAL_MODELS = {
    "0248_H_AOCERE_CAS": {
        "reference_url": BASE_URL + "svprojects/0248_H_AOCERE_CAS/0248_H_AOCERE_CAS_reference.zip",
        "all_url": BASE_URL + "svprojects/0248_H_AOCERE_CAS/0248_H_AOCERE_CAS_all.zip",
        "reference_variant": "40Left85Right",
    }
}


@dataclass
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    size: int
    header_offset: int


class RemoteZip:
    """Read individual entries of a remote zip file through HTTP Range requests."""

    def __init__(self, url: str, timeout: float = 120.0, retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        # Byte ranges refer to the raw file, so ask the server not to compress the transfer.
        self.session.headers["Accept-Encoding"] = "identity"
        self.size = self._remote_size()
        self.entries = self._read_central_directory()

    def _remote_size(self) -> int:
        head = self.session.head(self.url, timeout=self.timeout, allow_redirects=True)
        head.raise_for_status()
        if "Content-Length" in head.headers:
            return int(head.headers["Content-Length"])
        # Fallback: a one-byte range request reports the total size in Content-Range ("bytes 0-0/N").
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout)
        r.raise_for_status()
        content_range = r.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise IOError(f"cannot determine size of {self.url}")
        return int(content_range.rsplit("/", 1)[1])

    def _get(self, start: int, end: int) -> bytes:
        """Fetch bytes [start, end] inclusive."""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout
                )
                r.raise_for_status()
                if r.status_code != 206 and not (start == 0 and end == self.size - 1):
                    raise IOError(f"server ignored Range header (status {r.status_code})")
                return r.content
            except (requests.RequestException, IOError) as e:  # pragma: no cover - network
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        raise IOError(f"failed to fetch {self.url} [{start}-{end}]: {last_error}")

    def _read_central_directory(self) -> list[ZipEntry]:
        tail = self._get(max(0, self.size - 65536), self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise IOError("end of central directory not found")
        _, _, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[i : i + 22])
        if cd_offset == 0xFFFFFFFF or n_entries == 0xFFFF or cd_size == 0xFFFFFFFF:
            j = tail.rfind(b"PK\x06\x07")
            _, _, z64_offset, _ = struct.unpack("<IIQI", tail[j : j + 20])
            z64 = self._get(z64_offset, z64_offset + 55)
            (_, _, _, _, _, _, n_entries, _, cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", z64)
        cd = self._get(cd_offset, cd_offset + cd_size - 1)

        entries: list[ZipEntry] = []
        pos = 0
        while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
            fields = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
            method, csize, usize = fields[4], fields[8], fields[9]
            nlen, elen, clen, offset = fields[10], fields[11], fields[12], fields[16]
            name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
            extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
            if 0xFFFFFFFF in (csize, usize, offset):
                usize, csize, offset = _parse_zip64_extra(extra, usize, csize, offset)
            entries.append(ZipEntry(name, method, csize, usize, offset))
            pos += 46 + nlen + elen + clen
        return entries

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def read(self, entry: ZipEntry) -> bytes:
        header = self._get(entry.header_offset, entry.header_offset + 29)
        nlen, elen = struct.unpack("<HH", header[26:30])
        start = entry.header_offset + 30 + nlen + elen
        raw = self._get(start, start + entry.compressed_size - 1) if entry.compressed_size else b""
        if entry.method == 0:
            data = raw
        elif entry.method == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise IOError(f"unsupported compression method {entry.method} for {entry.name}")
        if len(data) != entry.size:
            raise IOError(f"size mismatch for {entry.name}: {len(data)} != {entry.size}")
        return data


def _parse_zip64_extra(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    p = 0
    while p + 4 <= len(extra):
        header_id, block_size = struct.unpack("<HH", extra[p : p + 4])
        p += 4
        if header_id == 0x0001:
            q = p
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", extra[q : q + 8])[0]
                q += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", extra[q : q + 8])[0]
                q += 8
            if offset == 0xFFFFFFFF:
                offset = struct.unpack("<Q", extra[q : q + 8])[0]
        p += block_size
    return usize, csize, offset


def relative_entry(name: str) -> str | None:
    """Strip the zip's root folder; return 'Paths/x.pth', 'Models/x.vtp', 'LICENSE', '*.pdf' or None."""
    if name.endswith("/"):
        return None
    parts = name.split("/")
    if len(parts) < 2:
        return None
    rel = parts[1:]
    if len(rel) == 1 and (rel[0] == "LICENSE" or rel[0].lower().endswith(".pdf")):
        return rel[0]
    if len(rel) == 2 and rel[0] == "Paths" and rel[1].endswith(".pth"):
        return "/".join(rel)
    if len(rel) == 2 and rel[0] == "Models" and rel[1].endswith(".vtp"):
        return "/".join(rel)
    return None


def default_variant(name: str) -> str | None:
    """Default surface-model variant for models that contain several (None for ordinary models)."""
    special = SPECIAL_MODELS.get(name)
    return special["reference_variant"] if special else None


def model_url(name: str, variant: str | None = None) -> str:
    special = SPECIAL_MODELS.get(name)
    if special is None:
        return f"{BASE_URL}svprojects/{name}.zip"
    if variant is None or variant == special["reference_variant"]:
        return special["reference_url"]
    return special["all_url"]


def fetch_model(
    name: str,
    variant: str | None = None,
    dest: Path | str = DEFAULT_RAW_DIR,
    verbose: bool = True,
) -> Path:
    """Download the centerlines, surface model(s) and license of one VMR model.

    Args:
        name: VMR model name, e.g. ``0248_H_AOCERE_CAS`` or ``0050_H_CERE_H``.
        variant: for models with several surface models (0248), which ``Models/{variant}.vtp`` to
            fetch. ``None`` fetches the default (reference variant for 0248, every .vtp otherwise).
        dest: root folder; files land in ``dest/{name}/``.

    Returns:
        The model folder.
    """
    out_dir = Path(dest) / name
    special = SPECIAL_MODELS.get(name)
    if special and variant is None:
        variant = special["reference_variant"]

    url = model_url(name, variant)
    if verbose:
        print(f"[fetch] {name}: reading zip index from {url}")
    rz = RemoteZip(url)

    wanted: list[tuple[ZipEntry, str]] = []
    for entry in rz.entries:
        rel = relative_entry(entry.name)
        if rel is None:
            continue
        if rel.startswith("Models/") and variant is not None and Path(rel).stem != variant:
            continue
        wanted.append((entry, rel))

    if not any(rel.startswith("Paths/") for _, rel in wanted):
        raise RuntimeError(f"{name}: no Paths/*.pth in the zip; this model has no centerlines")
    if variant is not None and not any(rel.startswith("Models/") for _, rel in wanted):
        available = sorted(
            {Path(e.name).stem for e in rz.entries if "/Models/" in e.name and e.name.endswith(".vtp")}
        )
        raise RuntimeError(f"{name}: variant {variant!r} not found. Available: {available}")

    total = 0
    for entry, rel in wanted:
        target = out_dir / rel
        if target.exists() and target.stat().st_size == entry.size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = rz.read(entry)
        target.write_bytes(data)
        total += len(data)
        if verbose:
            print(f"[fetch]   {rel} ({len(data) / 1e6:.2f} MB)")
    if verbose:
        print(f"[fetch] done: {out_dir} ({total / 1e6:.1f} MB downloaded)")
    return out_dir


def list_variants(name: str) -> list[str]:
    """List the surface-model variants in a model's zip (e.g. the 30 stenosis variants of 0248)."""
    special = SPECIAL_MODELS.get(name)
    url = special["all_url"] if special else model_url(name)
    rz = RemoteZip(url)
    return sorted(
        {Path(e.name).stem for e in rz.entries if "/Models/" in e.name and e.name.endswith(".vtp")}
    )


def load_index() -> list[dict[str, str]]:
    """Return the VMR model index as row dicts (columns: Name, Anatomy, Disease, Paths, Models, ...)."""
    r = requests.get(INDEX_URL, timeout=60)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
