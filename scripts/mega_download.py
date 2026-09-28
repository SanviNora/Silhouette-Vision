"""Download a public MEGA file link without the MEGA client.

Usage: python scripts/mega_download.py "<mega file url>" <output_dir> [--info]
"""

import base64
import json
import struct
import sys
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

API = "https://g.api.mega.co.nz/cs"
CHUNK = 1 << 20


def b64url_decode(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/").replace(",", "")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def parse_link(url: str) -> tuple[str, bytes]:
    # https://mega.nz/file/<id>#<key>
    path, key = url.split("#", 1)
    return path.rstrip("/").split("/")[-1], b64url_decode(key)


def split_key(raw: bytes) -> tuple[bytes, bytes]:
    k = struct.unpack(">8I", raw)
    aes_key = struct.pack(">4I", k[0] ^ k[4], k[1] ^ k[5], k[2] ^ k[6], k[3] ^ k[7])
    iv = struct.pack(">4I", k[4], k[5], 0, 0)
    return aes_key, iv


def file_info(file_id: str) -> dict:
    resp = requests.post(API, json=[{"a": "g", "g": 1, "p": file_id}], timeout=60)
    resp.raise_for_status()
    data = resp.json()[0]
    if isinstance(data, int):
        raise RuntimeError(f"MEGA API error code {data}")  # noqa: TRY004 (API status, not a type bug)
    return data


def decrypt_attrs(at: str, aes_key: bytes) -> dict:
    dec = Cipher(algorithms.AES(aes_key), modes.CBC(b"\0" * 16)).decryptor()
    raw = dec.update(b64url_decode(at)) + dec.finalize()
    text = raw.rstrip(b"\0").decode("utf-8", errors="ignore")
    return json.loads(text[4:]) if text.startswith("MEGA") else {}


def download(url: str, out_dir: Path, info_only: bool = False) -> Path:
    file_id, raw_key = parse_link(url)
    aes_key, iv = split_key(raw_key)
    info = file_info(file_id)
    name = decrypt_attrs(info["at"], aes_key).get("n", file_id)
    size = info["s"]
    print(f"{name}: {size / 1e9:.2f} GB")
    if info_only:
        return out_dir / name

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    done -= done % 16  # resume on an AES block boundary

    counter = int.from_bytes(iv, "big") + done // 16
    dec = Cipher(algorithms.AES(aes_key), modes.CTR(counter.to_bytes(16, "big"))).decryptor()
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(info["g"], headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "r+b" if done else "wb") as f:
            f.seek(done)
            f.truncate()
            for chunk in r.iter_content(CHUNK):
                f.write(dec.update(chunk))
                done += len(chunk)
                print(f"\r{done / size:6.1%}", end="", flush=True)
            f.write(dec.finalize())
    print()
    part.rename(target)
    return target


if __name__ == "__main__":
    download(sys.argv[1], Path(sys.argv[2]), info_only="--info" in sys.argv)
