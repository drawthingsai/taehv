import Foundation
import CoreML

enum ArgError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let msg):
            return msg
        }
    }
}

struct Config {
    let modelPath: String
    let iterations: Int
    let warmup: Int
    let computeUnits: MLComputeUnits
    let dtypeOverride: MLMultiArrayDataType?
    let seed: UInt64
}

struct LCG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func nextFloat(min: Float, max: Float) -> Float {
        state = 6364136223846793005 &* state &+ 1
        let x = Float((state >> 33) & 0xFFFF_FFFF) / Float(UInt32.max)
        return min + (max - min) * x
    }
}

func parseComputeUnits(_ value: String) throws -> MLComputeUnits {
    switch value {
    case "cpu_only":
        return .cpuOnly
    case "cpu_ne":
        return .cpuAndNeuralEngine
    case "cpu_gpu":
        return .cpuAndGPU
    case "all":
        return .all
    default:
        throw ArgError.message("Invalid --compute-units '\(value)'. Use cpu_only|cpu_ne|cpu_gpu|all.")
    }
}

func parseDtype(_ value: String) throws -> MLMultiArrayDataType {
    switch value {
    case "float16":
        return .float16
    case "float32":
        return .float32
    default:
        throw ArgError.message("Invalid --dtype '\(value)'. Use float16|float32.")
    }
}

func parseArgs() throws -> Config {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        throw ArgError.message(
            """
            Usage:
              benchmark_ltx2_decoder.swift <model.mlmodelc> [--iters N] [--warmup N] [--compute-units cpu_ne|all|cpu_only|cpu_gpu] [--dtype float16|float32] [--seed N]
            """
        )
    }

    var modelPath = args[1]
    var iterations = 100
    var warmup = 10
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var dtypeOverride: MLMultiArrayDataType?
    var seed: UInt64 = 0

    var i = 2
    while i < args.count {
        let key = args[i]
        switch key {
        case "--iters":
            i += 1
            guard i < args.count, let value = Int(args[i]), value > 0 else {
                throw ArgError.message("Invalid --iters value")
            }
            iterations = value
        case "--warmup":
            i += 1
            guard i < args.count, let value = Int(args[i]), value >= 0 else {
                throw ArgError.message("Invalid --warmup value")
            }
            warmup = value
        case "--compute-units":
            i += 1
            guard i < args.count else { throw ArgError.message("Missing --compute-units value") }
            computeUnits = try parseComputeUnits(args[i])
        case "--dtype":
            i += 1
            guard i < args.count else { throw ArgError.message("Missing --dtype value") }
            dtypeOverride = try parseDtype(args[i])
        case "--seed":
            i += 1
            guard i < args.count, let value = UInt64(args[i]) else {
                throw ArgError.message("Invalid --seed value")
            }
            seed = value
        default:
            if key.hasPrefix("--") {
                throw ArgError.message("Unknown option '\(key)'")
            } else {
                modelPath = key
            }
        }
        i += 1
    }

    return Config(
        modelPath: modelPath,
        iterations: iterations,
        warmup: warmup,
        computeUnits: computeUnits,
        dtypeOverride: dtypeOverride,
        seed: seed
    )
}

func shapeFromConstraint(_ constraint: MLMultiArrayConstraint) -> [Int] {
    return constraint.shape.map { $0.intValue }
}

func product(_ values: [Int]) -> Int {
    return values.reduce(1, *)
}

func fillInput(_ array: MLMultiArray, seed: UInt64) {
    var rng = LCG(seed: seed)
    let count = array.count
    switch array.dataType {
    case .float16:
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        for idx in 0..<count {
            ptr[idx] = Float16(rng.nextFloat(min: -1.0, max: 1.0))
        }
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for idx in 0..<count {
            ptr[idx] = rng.nextFloat(min: -1.0, max: 1.0)
        }
    default:
        break
    }
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let rank = p * Double(sorted.count - 1)
    let lo = Int(floor(rank))
    let hi = Int(ceil(rank))
    if lo == hi { return sorted[lo] }
    let t = rank - Double(lo)
    return sorted[lo] * (1.0 - t) + sorted[hi] * t
}

func run() throws {
    let cfg = try parseArgs()

    let modelURL = URL(fileURLWithPath: cfg.modelPath)
    let modelConfig = MLModelConfiguration()
    modelConfig.computeUnits = cfg.computeUnits
    let model = try MLModel(contentsOf: modelURL, configuration: modelConfig)

    guard let inputDesc = model.modelDescription.inputDescriptionsByName["latent"] else {
        throw ArgError.message("Model input 'latent' not found")
    }
    guard let inputConstraint = inputDesc.multiArrayConstraint else {
        throw ArgError.message("Input 'latent' is not MLMultiArray")
    }
    guard model.modelDescription.outputDescriptionsByName["image"] != nil else {
        throw ArgError.message("Model output 'image' not found")
    }

    let inputShape = shapeFromConstraint(inputConstraint)
    let inferredType = inputConstraint.dataType
    let inputType = cfg.dtypeOverride ?? inferredType
    if inputType != .float16 && inputType != .float32 {
        throw ArgError.message("Only float16/float32 input is supported, got \(inputType.rawValue)")
    }

    let inputArray = try MLMultiArray(
        shape: inputShape.map(NSNumber.init),
        dataType: inputType
    )
    fillInput(inputArray, seed: cfg.seed)

    let provider = try MLDictionaryFeatureProvider(dictionary: [
        "latent": MLFeatureValue(multiArray: inputArray)
    ])

    // Warmup run also validates output feature type/shape before timing.
    var outputShape: [Int] = []
    for i in 0..<cfg.warmup {
        let out = try model.prediction(from: provider)
        if i == 0 {
            guard let outArray = out.featureValue(for: "image")?.multiArrayValue else {
                throw ArgError.message("Output 'image' is not MLMultiArray")
            }
            outputShape = outArray.shape.map { $0.intValue }
        }
    }
    if outputShape.isEmpty {
        let out = try model.prediction(from: provider)
        guard let outArray = out.featureValue(for: "image")?.multiArrayValue else {
            throw ArgError.message("Output 'image' is not MLMultiArray")
        }
        outputShape = outArray.shape.map { $0.intValue }
    }

    // Timed runs
    var timingsMs: [Double] = []
    timingsMs.reserveCapacity(cfg.iterations)
    for _ in 0..<cfg.iterations {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: provider)
        let t1 = DispatchTime.now().uptimeNanoseconds
        timingsMs.append(Double(t1 - t0) / 1_000_000.0)
    }

    let mean = timingsMs.reduce(0.0, +) / Double(timingsMs.count)
    let median = percentile(timingsMs, 0.5)
    let p90 = percentile(timingsMs, 0.9)
    let p95 = percentile(timingsMs, 0.95)
    let minV = timingsMs.min() ?? 0.0
    let maxV = timingsMs.max() ?? 0.0

    let outputFrames = outputShape.count >= 2 ? outputShape[1] : 1
    let perFrameMs = mean / Double(max(outputFrames, 1))
    let fps = outputFrames > 0 ? (1000.0 * Double(outputFrames) / mean) : 0.0

    print("Model: \(cfg.modelPath)")
    print("ComputeUnits: \(cfg.computeUnits.rawValue)")
    print("Input shape: \(inputShape) (\(product(inputShape)) elems)")
    print("Output shape: \(outputShape)")
    print("Input dtype: \(inputType == .float16 ? "float16" : "float32")")
    print("Warmup: \(cfg.warmup), Iterations: \(cfg.iterations)")
    print(String(format: "Latency ms: mean=%.3f median=%.3f p90=%.3f p95=%.3f min=%.3f max=%.3f", mean, median, p90, p95, minV, maxV))
    print(String(format: "Temporal stats: output_frames=%d mean_ms_per_output_frame=%.3f mean_output_fps=%.2f", outputFrames, perFrameMs, fps))
}

do {
    try run()
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
