#!/usr/bin/env python3
"""Create/apply efficient binary patches for CoreML weight.bin files.

Patch format:
- Uses `git diff --binary -- weight.bin` from a temp git repo:
  - commit base `weight.bin`
  - overwrite with target `weight.bin`
  - export binary git patch
- Stores patch text + metadata JSON.
- Optional zstd compression for patch payload.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)


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

    with tempfile.TemporaryDirectory(prefix="weight_patch_create_") as td:
        tmp = Path(td)
        run(["git", "init", "-q"], cwd=tmp)
        shutil.copy2(base, tmp / "weight.bin")
        run(["git", "add", "weight.bin"], cwd=tmp)
        run(
            ["git", "-c", "user.name=patch", "-c", "user.email=patch@local", "commit", "-q", "-m", "base"],
            cwd=tmp,
        )
        shutil.copy2(target, tmp / "weight.bin")
        diff_proc = run(["git", "diff", "--binary", "--", "weight.bin"], cwd=tmp)
        patch_text = diff_proc.stdout
        if not patch_text.strip():
            raise SystemExit("Patch is empty (files may be identical).")

        raw_patch = tmp / "weight.patch"
        raw_patch.write_text(patch_text, encoding="utf-8")

        payload = raw_patch
        compression = "none"
        if compress == "zstd":
            zstd_bin = shutil.which("zstd")
            if zstd_bin is None:
                raise SystemExit("zstd not found in PATH; use --compress none or install zstd.")
            zstd_out = tmp / "weight.patch.zst"
            subprocess.run(
                [zstd_bin, "-q", f"-{zstd_level}", "-T0", str(raw_patch), "-o", str(zstd_out)],
                check=True,
            )
            payload = zstd_out
            compression = "zstd"

        patch_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payload, patch_out)

    metadata = {
        "format": "git-binary-patch-v1",
        "base_path_hint": str(base),
        "target_path_hint": str(target),
        "base_sha256": base_sha,
        "target_sha256": target_sha,
        "base_size": base.stat().st_size,
        "target_size": target.stat().st_size,
        "patch_file": patch_out.name,
        "compression": compression,
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

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

    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if metadata.get("format") != "git-binary-patch-v1":
        raise SystemExit(f"Unsupported patch format: {metadata.get('format')}")

    base_sha = sha256_file(base)
    expected_base_sha = metadata["base_sha256"]
    if base_sha != expected_base_sha:
        raise SystemExit(
            f"Base hash mismatch.\nExpected: {expected_base_sha}\nActual:   {base_sha}\nBase file: {base}"
        )

    with tempfile.TemporaryDirectory(prefix="weight_patch_apply_") as td:
        tmp = Path(td)
        run(["git", "init", "-q"], cwd=tmp)
        shutil.copy2(base, tmp / "weight.bin")
        run(["git", "add", "weight.bin"], cwd=tmp)
        run(
            ["git", "-c", "user.name=patch", "-c", "user.email=patch@local", "commit", "-q", "-m", "base"],
            cwd=tmp,
        )

        patch_to_apply = patch
        if metadata.get("compression") == "zstd":
            zstd_bin = shutil.which("zstd")
            if zstd_bin is None:
                raise SystemExit("zstd not found in PATH but patch is zstd-compressed.")
            decoded = tmp / "weight.patch"
            subprocess.run([zstd_bin, "-q", "-d", str(patch), "-o", str(decoded), "-f"], check=True)
            patch_to_apply = decoded

        subprocess.run(["git", "apply", "--binary", str(patch_to_apply)], cwd=str(tmp), check=True)
        target_file = tmp / "weight.bin"
        if not target_file.exists():
            raise SystemExit("Patch application succeeded but output weight.bin was not created.")

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_file, out)

    out_sha = sha256_file(out)
    expected_target_sha = metadata["target_sha256"]
    if out_sha != expected_target_sha:
        raise SystemExit(
            f"Output hash mismatch.\nExpected: {expected_target_sha}\nActual:   {out_sha}\nOutput: {out}"
        )

    print(f"Applied patch successfully.")
    print(f"Output:        {out}")
    print(f"Output sha256: {out_sha}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--base", required=True, help="Path to source weight.bin")
    c.add_argument("--target", required=True, help="Path to target weight.bin")
    c.add_argument("--patch-out", required=True, help="Output patch file (e.g. wan_stateful.patch.zst)")
    c.add_argument("--meta-out", required=True, help="Output metadata JSON")
    c.add_argument("--compress", choices=["none", "zstd"], default="zstd")
    c.add_argument("--zstd-level", type=int, default=19)

    a = sub.add_parser("apply")
    a.add_argument("--base", required=True, help="Path to source weight.bin")
    a.add_argument("--patch", required=True, help="Patch file path")
    a.add_argument("--meta", required=True, help="Metadata JSON path")
    a.add_argument("--out", required=True, help="Output reconstructed weight.bin path")

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
