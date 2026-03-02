import Foundation
import CryptoKit

enum InsdiffError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let msg):
            return msg
        }
    }
}

struct Config {
    let basePath: String
    let patchPath: String
    let outPath: String
}

struct PatchFile: Decodable {
    let format: String
    let base_sha256: String
    let target_sha256: String
    let base_size: Int
    let target_size: Int
    let ops: [PatchOp]
}

struct PatchOp: Decodable {
    let type: String
    let src_offset: Int?
    let payload_offset: Int?
    let length: Int
}

func parseArgs() throws -> Config {
    let args = CommandLine.arguments
    guard args.count >= 7 else {
        throw InsdiffError.message(
            """
            Usage:
              apply_weight_insdiff.swift --base <base.weight.bin> --patch <patch.insdiff> --out <output.weight.bin>
            """
        )
    }

    var basePath: String?
    var patchPath: String?
    var outPath: String?

    var i = 1
    while i < args.count {
        let key = args[i]
        switch key {
        case "--base":
            i += 1
            guard i < args.count else { throw InsdiffError.message("Missing --base value") }
            basePath = args[i]
        case "--patch":
            i += 1
            guard i < args.count else { throw InsdiffError.message("Missing --patch value") }
            patchPath = args[i]
        case "--out":
            i += 1
            guard i < args.count else { throw InsdiffError.message("Missing --out value") }
            outPath = args[i]
        default:
            throw InsdiffError.message("Unknown option: \(key)")
        }
        i += 1
    }

    guard let basePath, let patchPath, let outPath else {
        throw InsdiffError.message("Missing one of required options: --base --patch --out")
    }
    return Config(basePath: basePath, patchPath: patchPath, outPath: outPath)
}

func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func readPatch(_ path: String) throws -> (PatchFile, Data) {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard data.count >= 8 else {
        throw InsdiffError.message("Patch file too small: \(path)")
    }

    let metaLen = data.withUnsafeBytes { raw -> Int in
        let bytes = raw.bindMemory(to: UInt8.self)
        var v: UInt64 = 0
        for i in 0..<8 {
            v |= UInt64(bytes[i]) << UInt64(8 * i)
        }
        return Int(v)
    }

    guard metaLen >= 0, 8 + metaLen <= data.count else {
        throw InsdiffError.message("Invalid metadata length in patch: \(path)")
    }

    let metaData = data.subdata(in: 8..<(8 + metaLen))
    let payload = data.subdata(in: (8 + metaLen)..<data.count)

    let decoder = JSONDecoder()
    let meta = try decoder.decode(PatchFile.self, from: metaData)
    return (meta, payload)
}

func run() throws {
    let cfg = try parseArgs()
    let fm = FileManager.default

    guard fm.fileExists(atPath: cfg.basePath) else {
        throw InsdiffError.message("Base file not found: \(cfg.basePath)")
    }
    guard fm.fileExists(atPath: cfg.patchPath) else {
        throw InsdiffError.message("Patch file not found: \(cfg.patchPath)")
    }

    let baseData = try Data(contentsOf: URL(fileURLWithPath: cfg.basePath))
    let (patch, payload) = try readPatch(cfg.patchPath)

    guard patch.format == "insdiff-v1" else {
        throw InsdiffError.message("Unsupported patch format: \(patch.format)")
    }
    guard baseData.count == patch.base_size else {
        throw InsdiffError.message("Base size mismatch. expected=\(patch.base_size) actual=\(baseData.count)")
    }

    let baseHash = sha256Hex(baseData)
    guard baseHash == patch.base_sha256 else {
        throw InsdiffError.message("Base SHA256 mismatch. expected=\(patch.base_sha256) actual=\(baseHash)")
    }

    var out = Data()
    out.reserveCapacity(patch.target_size)

    for op in patch.ops {
        guard op.length >= 0 else {
            throw InsdiffError.message("Invalid op length: \(op.length)")
        }

        switch op.type {
        case "copy":
            guard let src = op.src_offset else {
                throw InsdiffError.message("copy op missing src_offset")
            }
            guard src >= 0, src + op.length <= baseData.count else {
                throw InsdiffError.message("copy out of range: src=\(src) len=\(op.length)")
            }
            out.append(baseData.subdata(in: src..<(src + op.length)))
        case "insert":
            guard let off = op.payload_offset else {
                throw InsdiffError.message("insert op missing payload_offset")
            }
            guard off >= 0, off + op.length <= payload.count else {
                throw InsdiffError.message("insert out of range: off=\(off) len=\(op.length)")
            }
            out.append(payload.subdata(in: off..<(off + op.length)))
        default:
            throw InsdiffError.message("Unknown op type: \(op.type)")
        }
    }

    guard out.count == patch.target_size else {
        throw InsdiffError.message("Output size mismatch. expected=\(patch.target_size) actual=\(out.count)")
    }

    let outHash = sha256Hex(out)
    guard outHash == patch.target_sha256 else {
        throw InsdiffError.message("Output SHA256 mismatch. expected=\(patch.target_sha256) actual=\(outHash)")
    }

    let outURL = URL(fileURLWithPath: cfg.outPath)
    let outDir = outURL.deletingLastPathComponent()
    if !outDir.path.isEmpty, !fm.fileExists(atPath: outDir.path) {
        try fm.createDirectory(at: outDir, withIntermediateDirectories: true)
    }
    try out.write(to: outURL, options: .atomic)

    print("Applied patch successfully.")
    print("Output: \(cfg.outPath)")
    print("SHA256: \(outHash)")
}

do {
    try run()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
