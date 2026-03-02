# INSDIFF

`insdiff` is a simple binary patch format for `weight.bin` dedup/sharing.

Goal:
- Keep runtime implementation easy in both Python and Swift.
- Avoid external binary-diff dependencies.
- Use deterministic reconstruction with SHA256 validation.

## Format (`insdiff-v1`)

Single-file container layout:
1. `uint64` little-endian: metadata JSON length
2. Metadata JSON bytes (UTF-8)
3. Raw payload bytes

Metadata schema:

```json
{
  "format": "insdiff-v1",
  "base_sha256": "...",
  "target_sha256": "...",
  "base_size": 15007488,
  "target_size": 19652352,
  "ops": [
    {"type": "copy", "src_offset": 0, "length": 123},
    {"type": "insert", "payload_offset": 0, "length": 456}
  ]
}
```

Operation semantics:
- `copy`: copy `length` bytes from `base[src_offset:src_offset+length]` into output.
- `insert`: copy `length` bytes from `payload[payload_offset:payload_offset+length]` into output.

Output bytes are produced by applying `ops` in order. Final output is accepted only if SHA256 equals `target_sha256`.

## Python implementation

File: `scripts/weight_bin_insdiff.py`

Implemented commands:
- `create`: build `insdiff` patch from `base` + `target`.
- `apply`: reconstruct target from `base` + patch.
- `inspect`: print patch stats (ops/copy/insert/payload/size).

Important details:
- Uses `sha256` checks for base and output validation.
- Uses a simple anchor/lookahead matcher to emit mostly `copy` ops and only required `insert` payload.
- Stores metadata + payload in a single `.insdiff` file.

Core apply logic:

```python
for op in meta["ops"]:
    t = op["type"]
    length = int(op["length"])
    if t == "copy":
        src = int(op["src_offset"])
        out_buf[w:w+length] = base_bytes[src:src+length]
    elif t == "insert":
        po = int(op["payload_offset"])
        out_buf[w:w+length] = payload[po:po+length]
    w += length
```

Example:

```bash
# Create patch
python3 scripts/weight_bin_insdiff.py create \
  --base qwenimage_tae/weight.bin \
  --target coreml_out/wan21_tae_firstframe_fullmem_build/768/wan21_decoder_firstframe_nhwc_fullmem.mlmodelc/weights/weight.bin \
  --patch-out wan_2.1_tae/weight.bin.insdiff

# Apply patch
python3 scripts/weight_bin_insdiff.py apply \
  --base qwenimage_tae/weight.bin \
  --patch wan_2.1_tae/weight.bin.insdiff \
  --out /tmp/wan21_weight.bin
```

## Swift implementation

File: `scripts/apply_weight_insdiff.swift`

Implemented behavior:
- Parse args: `--base`, `--patch`, `--out`.
- Parse patch container (`uint64` LE metadata length, JSON metadata, payload).
- Validate:
  - `format == insdiff-v1`
  - `base_size` matches
  - `base_sha256` matches (`CryptoKit`)
- Apply operations in order (`copy` / `insert`) into output buffer.
- Validate output SHA256 equals `target_sha256`.
- Write output atomically.

Core apply logic:

```swift
for op in patch.ops {
    switch op.type {
    case "copy":
        let src = op.src_offset!
        out.append(baseData.subdata(in: src..<(src + op.length)))
    case "insert":
        let off = op.payload_offset!
        out.append(payload.subdata(in: off..<(off + op.length)))
    default:
        throw InsdiffError.message("Unknown op type: \(op.type)")
    }
}
```

Build + run:

```bash
CLANG_MODULE_CACHE_PATH=/tmp/clang-module-cache SWIFTPM_CACHE_PATH=/tmp/swiftpm-cache \
swiftc scripts/apply_weight_insdiff.swift -O -o /tmp/apply_weight_insdiff

/tmp/apply_weight_insdiff \
  --base qwenimage_tae/weight.bin \
  --patch wan_2.1_tae/weight.bin.insdiff \
  --out /tmp/wan21_weight.bin
```

## Current usage in this repo

For `wan_2.1_tae` packaging:
- Per-size `.mlmodelc` bundles do **not** contain `weights/weight.bin`.
- A shared `wan_2.1_tae/weight.bin.insdiff` is shipped instead.
- Runtime reconstructs Wan 2.1 weight from `qwenimage_tae/weight.bin` + insdiff.

Verified reconstruction hash:
- `9051de85ffdb16b6d647a904e0a05bb9b007cd73150e9af9e5c77f05cd5f19bd`
