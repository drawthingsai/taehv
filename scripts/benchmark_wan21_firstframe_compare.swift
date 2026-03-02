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
    let beforeModelPath: String
    let afterModelPath: String
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

struct TimingStats {
    let mean: Double
    let median: Double
    let p90: Double
    let p95: Double
    let minV: Double
    let maxV: Double
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
    guard args.count >= 3 else {
        throw ArgError.message(
            """
            Usage:
              benchmark_wan21_firstframe_compare.swift <before.mlmodelc> <after.mlmodelc> [--iters N] [--warmup N] [--compute-units cpu_ne|all|cpu_only|cpu_gpu] [--dtype float16|float32] [--seed N]
            """
        )
    }

    let beforeModelPath = args[1]
    let afterModelPath = args[2]
    var iterations = 200
    var warmup = 20
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var dtypeOverride: MLMultiArrayDataType?
    var seed: UInt64 = 0

    var i = 3
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
            throw ArgError.message("Unknown option '\(key)'")
        }
        i += 1
    }

    return Config(
        beforeModelPath: beforeModelPath,
        afterModelPath: afterModelPath,
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
        for i in 0..<count {
            ptr[i] = Float16(rng.nextFloat(min: -1.0, max: 1.0))
        }
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = rng.nextFloat(min: -1.0, max: 1.0)
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

func summarize(_ timingsMs: [Double]) -> TimingStats {
    let mean = timingsMs.reduce(0.0, +) / Double(timingsMs.count)
    let median = percentile(timingsMs, 0.5)
    let p90 = percentile(timingsMs, 0.9)
    let p95 = percentile(timingsMs, 0.95)
    let minV = timingsMs.min() ?? 0.0
    let maxV = timingsMs.max() ?? 0.0
    return TimingStats(mean: mean, median: median, p90: p90, p95: p95, minV: minV, maxV: maxV)
}

func benchmarkModel(
    _ model: MLModel,
    provider: MLFeatureProvider,
    warmup: Int,
    iterations: Int
) throws -> TimingStats {
    for _ in 0..<warmup {
        _ = try model.prediction(from: provider)
    }

    var timingsMs: [Double] = []
    timingsMs.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: provider)
        let t1 = DispatchTime.now().uptimeNanoseconds
        timingsMs.append(Double(t1 - t0) / 1_000_000.0)
    }
    return summarize(timingsMs)
}

func run() throws {
    let cfg = try parseArgs()

    let modelConfig = MLModelConfiguration()
    modelConfig.computeUnits = cfg.computeUnits

    let beforeModel = try MLModel(
        contentsOf: URL(fileURLWithPath: cfg.beforeModelPath),
        configuration: modelConfig
    )
    let afterModel = try MLModel(
        contentsOf: URL(fileURLWithPath: cfg.afterModelPath),
        configuration: modelConfig
    )

    guard let beforeInputDesc = beforeModel.modelDescription.inputDescriptionsByName["latent"],
          let beforeConstraint = beforeInputDesc.multiArrayConstraint else {
        throw ArgError.message("Before model input 'latent' not found or not MLMultiArray")
    }
    guard let afterInputDesc = afterModel.modelDescription.inputDescriptionsByName["latent"],
          let afterConstraint = afterInputDesc.multiArrayConstraint else {
        throw ArgError.message("After model input 'latent' not found or not MLMultiArray")
    }

    let beforeShape = shapeFromConstraint(beforeConstraint)
    let afterShape = shapeFromConstraint(afterConstraint)
    if beforeShape != afterShape {
        throw ArgError.message("Input shapes differ: before=\(beforeShape), after=\(afterShape)")
    }

    let inputType = cfg.dtypeOverride ?? beforeConstraint.dataType
    if inputType != .float16 && inputType != .float32 {
        throw ArgError.message("Only float16/float32 input is supported, got \(inputType.rawValue)")
    }

    let inputArray = try MLMultiArray(
        shape: beforeShape.map(NSNumber.init),
        dataType: inputType
    )
    fillInput(inputArray, seed: cfg.seed)
    let provider = try MLDictionaryFeatureProvider(dictionary: [
        "latent": MLFeatureValue(multiArray: inputArray)
    ])

    let beforeStats = try benchmarkModel(
        beforeModel,
        provider: provider,
        warmup: cfg.warmup,
        iterations: cfg.iterations
    )
    let afterStats = try benchmarkModel(
        afterModel,
        provider: provider,
        warmup: cfg.warmup,
        iterations: cfg.iterations
    )

    let deltaMean = afterStats.mean - beforeStats.mean
    let deltaMedian = afterStats.median - beforeStats.median
    let deltaP95 = afterStats.p95 - beforeStats.p95
    let pctMean = beforeStats.mean != 0 ? (deltaMean / beforeStats.mean * 100.0) : 0.0
    let pctMedian = beforeStats.median != 0 ? (deltaMedian / beforeStats.median * 100.0) : 0.0
    let pctP95 = beforeStats.p95 != 0 ? (deltaP95 / beforeStats.p95 * 100.0) : 0.0

    print("ComputeUnits: \(cfg.computeUnits.rawValue)")
    print("Input shape: \(beforeShape) (\(product(beforeShape)) elems)")
    print("Input dtype: \(inputType == .float16 ? "float16" : "float32")")
    print("Warmup: \(cfg.warmup), Iterations: \(cfg.iterations)")
    print("")
    print("Before model: \(cfg.beforeModelPath)")
    print(String(format: "Latency ms: mean=%.3f median=%.3f p90=%.3f p95=%.3f min=%.3f max=%.3f", beforeStats.mean, beforeStats.median, beforeStats.p90, beforeStats.p95, beforeStats.minV, beforeStats.maxV))
    print("")
    print("After model: \(cfg.afterModelPath)")
    print(String(format: "Latency ms: mean=%.3f median=%.3f p90=%.3f p95=%.3f min=%.3f max=%.3f", afterStats.mean, afterStats.median, afterStats.p90, afterStats.p95, afterStats.minV, afterStats.maxV))
    print("")
    print("Delta (after - before):")
    print(String(format: "mean: %+0.3f ms (%+0.2f%%)", deltaMean, pctMean))
    print(String(format: "median: %+0.3f ms (%+0.2f%%)", deltaMedian, pctMedian))
    print(String(format: "p95: %+0.3f ms (%+0.2f%%)", deltaP95, pctP95))
}

do {
    try run()
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
