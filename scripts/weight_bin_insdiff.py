#!/usr/bin/env python3
"""Create/apply a simple copy+insert patch format for weight.bin.

Container layout (single file):
- uint64 little-endian JSON metadata length
- metadata JSON bytes (UTF-8)
- payload bytes

Metadata schema:
{
  "format": "insdiff-v1",
  "base_sha256": "...",
  "target_sha256": "...",
  "base_size": 0,
  "target_size": 0,
  "ops": [
    {"type": "copy", "src_offset": 0, "length": 123},
    {"type": "insert", "payload_offset": 0, "length": 456}
  ]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpCopy:
    src_offset: int
    length: int


@dataclass
class OpInsert:
    payload_offset: int
    length: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def append_copy(ops: list[dict], src_offset: int, length: int) -> None:
    if length <= 0:
        return
    if ops and ops[-1]["type"] == "copy":
        last = ops[-1]
        last_end = int(last["src_offset"]) + int(last["length"])
        if last_end == src_offset:
            last["length"] = int(last["length"]) + length
            return
    ops.append({"type": "copy", "src_offset": src_offset, "length": length})


def append_insert(ops: list[dict], payload: bytearray, chunk: bytes) -> None:
    if not chunk:
        return
    off = len(payload)
    payload.extend(chunk)
    if ops and ops[-1]["type"] == "insert":
        last = ops[-1]
        last_end = int(last["payload_offset"]) + int(last["length"])
        if last_end == off:
            last["length"] = int(last["length"]) + len(chunk)
            return
    ops.append({"type": "insert", "payload_offset": off, "length": len(chunk)})


def build_ops(
    base: bytes,
    target: bytes,
    anchor_bytes: int,
    lookahead_target: int,
    lookahead_base: int,
) -> tuple[list[dict], bytes]:
    m = len(base)
    n = len(target)
    b = 0
    i = 0

    ops: list[dict] = []
    payload = bytearray()
    pending_insert = bytearray()

    def flush_pending() -> None:
        nonlocal pending_insert
        if pending_insert:
            append_insert(ops, payload, bytes(pending_insert))
            pending_insert = bytearray()

    while i < n:
        if b < m and base[b] == target[i]:
            j = i
            k = b
            while j < n and k < m and target[j] == base[k]:
                j += 1
                k += 1
            match_len = j - i
            if match_len >= anchor_bytes:
                flush_pending()
                append_copy(ops, b, match_len)
                i = j
                b = k
                continue

        if b >= m:
            pending_insert.extend(target[i:])
            i = n
            break

        inserted = False
        if b + anchor_bytes <= m:
            base_anchor = base[b : b + anchor_bytes]
            t_end = min(n, i + lookahead_target)
            pos_t = target.find(base_anchor, i + 1, t_end)
            if pos_t != -1:
                pending_insert.extend(target[i:pos_t])
                i = pos_t
                inserted = True
        if inserted:
            continue

        if i + anchor_bytes <= n:
            target_anchor = target[i : i + anchor_bytes]
            b_end = min(m, b + lookahead_base)
            pos_b = base.find(target_anchor, b + 1, b_end)
            if pos_b != -1:
                b = pos_b
                continue

        pending_insert.append(target[i])
        i += 1

    flush_pending()
    return ops, bytes(payload)


def write_patch(path: Path, metadata: dict, payload: bytes) -> None:
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(payload)


def read_patch(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 8:
        raise SystemExit(f"Patch file too small: {path}")
    meta_len = struct.unpack("<Q", data[:8])[0]
    if 8 + meta_len > len(data):
        raise SystemExit(f"Invalid metadata length in patch: {path}")
    meta = json.loads(data[8 : 8 + meta_len].decode("utf-8"))
    payload = data[8 + meta_len :]
    return meta, payload


def create_patch(
    base: Path,
    target: Path,
    patch_out: Path,
    anchor_bytes: int,
    lookahead_target: int,
    lookahead_base: int,
) -> None:
    if not base.exists():
        raise SystemExit(f"Base file not found: {base}")
    if not target.exists():
        raise SystemExit(f"Target file not found: {target}")

    base_bytes = base.read_bytes()
    target_bytes = target.read_bytes()

    ops, payload = build_ops(
        base=base_bytes,
        target=target_bytes,
        anchor_bytes=anchor_bytes,
        lookahead_target=lookahead_target,
        lookahead_base=lookahead_base,
    )

    meta = {
        "format": "insdiff-v1",
        "base_sha256": sha256_bytes(base_bytes),
        "target_sha256": sha256_bytes(target_bytes),
        "base_size": len(base_bytes),
        "target_size": len(target_bytes),
        "anchor_bytes": anchor_bytes,
        "lookahead_target": lookahead_target,
        "lookahead_base": lookahead_base,
        "ops": ops,
    }

    write_patch(patch_out, meta, payload)

    copy_bytes = sum(int(op["length"]) for op in ops if op["type"] == "copy")
    insert_bytes = sum(int(op["length"]) for op in ops if op["type"] == "insert")

    print(f"Base sha256:   {meta['base_sha256']}")
    print(f"Target sha256: {meta['target_sha256']}")
    print(f"Ops:           {len(ops)} (copy={copy_bytes} insert={insert_bytes})")
    print(f"Payload:       {len(payload)} bytes")
    print(f"Patch:         {patch_out} ({patch_out.stat().st_size} bytes)")


def apply_patch(base: Path, patch: Path, out: Path) -> None:
    if not base.exists():
        raise SystemExit(f"Base file not found: {base}")
    if not patch.exists():
        raise SystemExit(f"Patch file not found: {patch}")

    base_bytes = base.read_bytes()
    meta, payload = read_patch(patch)

    if meta.get("format") != "insdiff-v1":
        raise SystemExit(f"Unsupported patch format: {meta.get('format')}")

    base_sha = sha256_bytes(base_bytes)
    if base_sha != meta["base_sha256"]:
        raise SystemExit(
            f"Base hash mismatch.\nExpected: {meta['base_sha256']}\nActual:   {base_sha}\nBase: {base}"
        )

    out_buf = bytearray()
    out_buf.extend(b"\x00" * int(meta["target_size"]))
    w = 0

    for op in meta["ops"]:
        t = op["type"]
        length = int(op["length"])
        if length < 0:
            raise SystemExit(f"Invalid op length: {length}")
        if t == "copy":
            src = int(op["src_offset"])
            if src < 0 or src + length > len(base_bytes):
                raise SystemExit(f"Invalid copy range: src={src} len={length}")
            out_buf[w : w + length] = base_bytes[src : src + length]
        elif t == "insert":
            po = int(op["payload_offset"])
            if po < 0 or po + length > len(payload):
                raise SystemExit(f"Invalid insert range: off={po} len={length}")
            out_buf[w : w + length] = payload[po : po + length]
        else:
            raise SystemExit(f"Unknown op type: {t}")
        w += length

    if w != len(out_buf):
        raise SystemExit(f"Output length mismatch while applying ops: wrote={w} expected={len(out_buf)}")

    out_sha = sha256_bytes(bytes(out_buf))
    if out_sha != meta["target_sha256"]:
        raise SystemExit(
            f"Output hash mismatch.\nExpected: {meta['target_sha256']}\nActual:   {out_sha}\nOut: {out}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(out_buf)
    print("Applied patch successfully.")
    print(f"Output:        {out}")
    print(f"Output sha256: {out_sha}")


def inspect_patch(patch: Path) -> None:
    if not patch.exists():
        raise SystemExit(f"Patch file not found: {patch}")
    meta, payload = read_patch(patch)
    copy_bytes = sum(int(op["length"]) for op in meta.get("ops", []) if op.get("type") == "copy")
    insert_bytes = sum(int(op["length"]) for op in meta.get("ops", []) if op.get("type") == "insert")

    print(json.dumps(
        {
            "format": meta.get("format"),
            "base_sha256": meta.get("base_sha256"),
            "target_sha256": meta.get("target_sha256"),
            "base_size": meta.get("base_size"),
            "target_size": meta.get("target_size"),
            "ops": len(meta.get("ops", [])),
            "copy_bytes": copy_bytes,
            "insert_bytes": insert_bytes,
            "payload_size": len(payload),
            "patch_size": patch.stat().st_size,
        },
        indent=2,
    ))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--base", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--patch-out", required=True)
    c.add_argument("--anchor-bytes", type=int, default=32)
    c.add_argument("--lookahead-target", type=int, default=1_048_576)
    c.add_argument("--lookahead-base", type=int, default=1_048_576)

    a = sub.add_parser("apply")
    a.add_argument("--base", required=True)
    a.add_argument("--patch", required=True)
    a.add_argument("--out", required=True)

    i = sub.add_parser("inspect")
    i.add_argument("--patch", required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "create":
        create_patch(
            base=Path(args.base),
            target=Path(args.target),
            patch_out=Path(args.patch_out),
            anchor_bytes=args.anchor_bytes,
            lookahead_target=args.lookahead_target,
            lookahead_base=args.lookahead_base,
        )
    elif args.cmd == "apply":
        apply_patch(
            base=Path(args.base),
            patch=Path(args.patch),
            out=Path(args.out),
        )
    elif args.cmd == "inspect":
        inspect_patch(Path(args.patch))
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
