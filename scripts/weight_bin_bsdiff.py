#!/usr/bin/env python3
"""Create/apply BSDIFF patches for CoreML weight.bin with hash validation.

Requires:
- Python package: bsdiff4 (already installed in _env)
- System tool for apply path: /usr/bin/bspatch
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import bsdiff4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def create_patch(
    base: Path,
    target: Path,
    patch_out: Path,
    meta_out: Path,
    compress: str,
    zstd_level: int,
) -> None:
    base = base.resolve()
    target = target.resolve()
    patch_out = patch_out.resolve()
    meta_out = meta_out.resolve()

    if not base.exists():
        raise SystemExit(f"Base file not found: {base}")
    if not target.exists():
        raise SystemExit(f"Target file not found: {target}")

    base_sha = sha256_file(base)
    target_sha = sha256_file(target)

    with tempfile.TemporaryDirectory(prefix="wan_bsdiff_create_") as td:
        tmp = Path(td)
        raw_patch = tmp / "weight.bsdiff"
        bsdiff4.file_diff(str(base), str(target), str(raw_patch))

        payload = raw_patch
        compression = "none"
        if compress == "zstd":
            zstd_bin = shutil.which("zstd")
            if zstd_bin is None:
                raise SystemExit("zstd not found in PATH; use --compress none or install zstd.")
            zstd_out = tmp / "weight.bsdiff.zst"
            subprocess.run([zstd_bin, "-q", f"-{zstd_level}", "-T0", str(raw_patch), "-o", str(zstd_out)], check=True)
            payload = zstd_out
            compression = "zstd"

        patch_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payload, patch_out)

    meta = {
        "format": "bsdiff-v1",
        "compression": compression,
        "base_path_hint": str(base),
        "target_path_hint": str(target),
        "base_sha256": base_sha,
        "target_sha256": target_sha,
        "base_size": base.stat().st_size,
        "target_size": target.stat().st_size,
        "patch_file": patch_out.name,
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Base sha256:   {base_sha}")
    print(f"Target sha256: {target_sha}")
    print(f"Wrote patch:   {patch_out} ({patch_out.stat().st_size} bytes)")
    print(f"Wrote meta:    {meta_out}")


def apply_patch(
    base: Path,
    patch: Path,
    meta: Path,
    out: Path,
) -> None:
    base = base.resolve()
    patch = patch.resolve()
    meta = meta.resolve()
    out = out.resolve()

    if not base.exists():
        raise SystemExit(f"Base file not found: {base}")
    if not patch.exists():
        raise SystemExit(f"Patch file not found: {patch}")
    if not meta.exists():
        raise SystemExit(f"Metadata file not found: {meta}")

    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("format") != "bsdiff-v1":
        raise SystemExit(f"Unsupported patch format: {info.get('format')}")

    base_sha = sha256_file(base)
    if base_sha != info["base_sha256"]:
        raise SystemExit(
            f"Base hash mismatch.\nExpected: {info['base_sha256']}\nActual:   {base_sha}\nBase: {base}"
        )

    with tempfile.TemporaryDirectory(prefix="wan_bsdiff_apply_") as td:
        tmp = Path(td)
        patch_to_apply = patch
        if info.get("compression") == "zstd":
            zstd_bin = shutil.which("zstd")
            if zstd_bin is None:
                raise SystemExit("zstd not found in PATH but patch is zstd-compressed.")
            decoded = tmp / "weight.bsdiff"
            subprocess.run([zstd_bin, "-q", "-d", str(patch), "-o", str(decoded), "-f"], check=True)
            patch_to_apply = decoded

        bspatch = Path("/usr/bin/bspatch")
        if not bspatch.exists():
            raise SystemExit("/usr/bin/bspatch not found on this machine.")

        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(bspatch), str(base), str(out), str(patch_to_apply)], check=True)

    out_sha = sha256_file(out)
    if out_sha != info["target_sha256"]:
        raise SystemExit(
            f"Output hash mismatch.\nExpected: {info['target_sha256']}\nActual:   {out_sha}\nOut: {out}"
        )

    print("Applied patch successfully.")
    print(f"Output:        {out}")
    print(f"Output sha256: {out_sha}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--base", required=True, help="Path to source weight.bin")
    c.add_argument("--target", required=True, help="Path to target weight.bin")
    c.add_argument("--patch-out", required=True, help="Patch file output")
    c.add_argument("--meta-out", required=True, help="Metadata JSON output")
    c.add_argument("--compress", choices=["none", "zstd"], default="none")
    c.add_argument("--zstd-level", type=int, default=19)

    a = sub.add_parser("apply")
    a.add_argument("--base", required=True, help="Path to source weight.bin")
    a.add_argument("--patch", required=True, help="Patch file path")
    a.add_argument("--meta", required=True, help="Metadata JSON path")
    a.add_argument("--out", required=True, help="Reconstructed weight.bin output")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "create":
        create_patch(
            base=Path(args.base),
            target=Path(args.target),
            patch_out=Path(args.patch_out),
            meta_out=Path(args.meta_out),
            compress=args.compress,
            zstd_level=args.zstd_level,
        )
    elif args.cmd == "apply":
        apply_patch(
            base=Path(args.base),
            patch=Path(args.patch),
            meta=Path(args.meta),
            out=Path(args.out),
        )
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
