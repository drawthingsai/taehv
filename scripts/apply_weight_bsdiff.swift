import Foundation
import CryptoKit

enum PatchError: Error, CustomStringConvertible {
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
    let metaPath: String
    let outPath: String
    let bspatchPath: String
}

struct PatchMetadata: Decodable {
    let format: String
    let compression: String
    let base_sha256: String
    let target_sha256: String
}

func parseArgs() throws -> Config {
    let args = CommandLine.arguments
    guard args.count >= 9 else {
        throw PatchError.message(
            """
            Usage:
              apply_weight_bsdiff.swift \\
                --base <base.weight.bin> \\
                --patch <patch.bsdiff> \\
                --meta <patch.json> \\
                --out <output.weight.bin> \\
                [--bspatch /usr/bin/bspatch]
            """
        )
    }

    var basePath: String?
    var patchPath: String?
    var metaPath: String?
    var outPath: String?
    var bspatchPath = "/usr/bin/bspatch"

    var i = 1
    while i < args.count {
        let key = args[i]
        switch key {
        case "--base":
            i += 1
            guard i < args.count else { throw PatchError.message("Missing --base value") }
            basePath = args[i]
        case "--patch":
            i += 1
            guard i < args.count else { throw PatchError.message("Missing --patch value") }
            patchPath = args[i]
        case "--meta":
            i += 1
            guard i < args.count else { throw PatchError.message("Missing --meta value") }
            metaPath = args[i]
        case "--out":
            i += 1
            guard i < args.count else { throw PatchError.message("Missing --out value") }
            outPath = args[i]
        case "--bspatch":
            i += 1
            guard i < args.count else { throw PatchError.message("Missing --bspatch value") }
            bspatchPath = args[i]
        default:
            throw PatchError.message("Unknown option: \(key)")
        }
        i += 1
    }

    guard let basePath, let patchPath, let metaPath, let outPath else {
        throw PatchError.message("Missing one of required options: --base --patch --meta --out")
    }

    return Config(
        basePath: basePath,
        patchPath: patchPath,
        metaPath: metaPath,
        outPath: outPath,
        bspatchPath: bspatchPath
    )
}

func sha256Hex(url: URL) throws -> String {
    guard let input = InputStream(url: url) else {
        throw PatchError.message("Unable to open stream for \(url.path)")
    }
    input.open()
    defer { input.close() }

    var hasher = SHA256()
    let bufferSize = 1024 * 1024
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
    defer { buffer.deallocate() }

    while true {
        let read = input.read(buffer, maxLength: bufferSize)
        if read < 0 {
            throw input.streamError ?? PatchError.message("Failed to read \(url.path)")
        }
        if read == 0 {
            break
        }
        hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: read))
    }

    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

func loadMetadata(path: String) throws -> PatchMetadata {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decoder = JSONDecoder()
    return try decoder.decode(PatchMetadata.self, from: data)
}

func runProcess(executable: String, args: [String]) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = args

    let stderrPipe = Pipe()
    process.standardError = stderrPipe

    try process.run()
    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let errText = String(data: errData, encoding: .utf8) ?? ""
        throw PatchError.message(
            "bspatch failed (status \(process.terminationStatus)): \(errText.trimmingCharacters(in: .whitespacesAndNewlines))"
        )
    }
}

func run() throws {
    let cfg = try parseArgs()
    let fm = FileManager.default

    guard fm.fileExists(atPath: cfg.basePath) else {
        throw PatchError.message("Base file not found: \(cfg.basePath)")
    }
    guard fm.fileExists(atPath: cfg.patchPath) else {
        throw PatchError.message("Patch file not found: \(cfg.patchPath)")
    }
    guard fm.fileExists(atPath: cfg.metaPath) else {
        throw PatchError.message("Metadata file not found: \(cfg.metaPath)")
    }
    guard fm.isExecutableFile(atPath: cfg.bspatchPath) else {
        throw PatchError.message("bspatch not executable: \(cfg.bspatchPath)")
    }

    let meta = try loadMetadata(path: cfg.metaPath)
    guard meta.format == "bsdiff-v1" else {
        throw PatchError.message("Unsupported patch format: \(meta.format)")
    }
    guard meta.compression == "none" else {
        throw PatchError.message("Unsupported compression '\(meta.compression)'. Use uncompressed patches for Swift path.")
    }

    let baseURL = URL(fileURLWithPath: cfg.basePath)
    let outURL = URL(fileURLWithPath: cfg.outPath)

    let baseHash = try sha256Hex(url: baseURL)
    guard baseHash == meta.base_sha256 else {
        throw PatchError.message("Base SHA256 mismatch. expected=\(meta.base_sha256) actual=\(baseHash)")
    }

    let outDir = outURL.deletingLastPathComponent()
    if !outDir.path.isEmpty, !fm.fileExists(atPath: outDir.path) {
        try fm.createDirectory(at: outDir, withIntermediateDirectories: true)
    }
    if fm.fileExists(atPath: outURL.path) {
        try fm.removeItem(at: outURL)
    }

    try runProcess(executable: cfg.bspatchPath, args: [cfg.basePath, cfg.outPath, cfg.patchPath])

    let outHash = try sha256Hex(url: outURL)
    guard outHash == meta.target_sha256 else {
        throw PatchError.message("Output SHA256 mismatch. expected=\(meta.target_sha256) actual=\(outHash)")
    }

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
