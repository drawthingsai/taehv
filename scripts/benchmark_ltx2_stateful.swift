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
    let statefulModelPath: String
    let referenceModelPath: String?
    let lengths: [Int]
    let warmup: Int
    let iterations: Int
    let trimFrames: Int
    let computeUnits: MLComputeUnits
    let dtypeOverride: MLMultiArrayDataType?
    let seed: UInt64
    let checkCorrectness: Bool
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

struct DiffStats {
    var sumAbs: Double = 0
    var maxAbs: Float = 0
    var count: Int = 0

    mutating func add(diff: Float) {
        let d = abs(diff)
        sumAbs += Double(d)
        if d > maxAbs { maxAbs = d }
        count += 1
    }

    var mae: Double {
        guard count > 0 else { return 0 }
        return sumAbs / Double(count)
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

func parseIntList(_ value: String) throws -> [Int] {
    let parts = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    let out = parts.compactMap(Int.init)
    if out.isEmpty || out.count != parts.count {
        throw ArgError.message("Invalid --lengths '\(value)'. Expected comma-separated integers.")
    }
    return out
}

func parseArgs() throws -> Config {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        throw ArgError.message(
            """
            Usage:
              benchmark_ltx2_stateful.swift <stateful.mlmodelc> [--reference-model <oneshot.mlmodelc>] [--lengths 1,2,3,5,7,11] [--warmup N] [--iters N] [--trim 7] [--compute-units cpu_ne|all|cpu_only|cpu_gpu] [--dtype float16|float32] [--seed N] [--no-correctness]
            """
        )
    }

    var statefulModelPath = args[1]
    var referenceModelPath: String?
    var lengths = [1, 2, 3, 5, 7, 11]
    var warmup = 3
    var iterations = 20
    var trimFrames = 7
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var dtypeOverride: MLMultiArrayDataType?
    var seed: UInt64 = 0
    var checkCorrectness = true

    var i = 2
    while i < args.count {
        let key = args[i]
        switch key {
        case "--reference-model":
            i += 1
            guard i < args.count else { throw ArgError.message("Missing --reference-model value") }
            referenceModelPath = args[i]
        case "--lengths":
            i += 1
            guard i < args.count else { throw ArgError.message("Missing --lengths value") }
            lengths = try parseIntList(args[i])
        case "--warmup":
            i += 1
            guard i < args.count, let value = Int(args[i]), value >= 0 else {
                throw ArgError.message("Invalid --warmup value")
            }
            warmup = value
        case "--iters":
            i += 1
            guard i < args.count, let value = Int(args[i]), value > 0 else {
                throw ArgError.message("Invalid --iters value")
            }
            iterations = value
        case "--trim":
            i += 1
            guard i < args.count, let value = Int(args[i]), value >= 0 else {
                throw ArgError.message("Invalid --trim value")
            }
            trimFrames = value
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
        case "--no-correctness":
            checkCorrectness = false
        default:
            if key.hasPrefix("--") {
                throw ArgError.message("Unknown option '\(key)'")
            } else {
                statefulModelPath = key
            }
        }
        i += 1
    }

    if checkCorrectness && referenceModelPath == nil {
        throw ArgError.message("Correctness check requires --reference-model.")
    }

    return Config(
        statefulModelPath: statefulModelPath,
        referenceModelPath: referenceModelPath,
        lengths: lengths,
        warmup: warmup,
        iterations: iterations,
        trimFrames: trimFrames,
        computeUnits: computeUnits,
        dtypeOverride: dtypeOverride,
        seed: seed,
        checkCorrectness: checkCorrectness
    )
}

func shapeFromConstraint(_ constraint: MLMultiArrayConstraint) -> [Int] {
    constraint.shape.map { $0.intValue }
}

func parseActIndex(_ name: String) -> Int? {
    if name.hasPrefix("act_"), name.hasSuffix("_out") {
        let middle = name.dropFirst("act_".count).dropLast("_out".count)
        return Int(middle)
    }
    if name.hasPrefix("act_") {
        let middle = name.dropFirst("act_".count)
        return Int(middle)
    }
    return nil
}

func sortedActNames(_ names: [String], isOutput: Bool) -> [String] {
    names
        .filter { name in
            if isOutput { return name.hasPrefix("act_") && name.hasSuffix("_out") }
            return name.hasPrefix("act_") && !name.hasSuffix("_out")
        }
        .sorted { (lhs, rhs) in
            (parseActIndex(lhs) ?? Int.max) < (parseActIndex(rhs) ?? Int.max)
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

func makeMultiArray(shape: [Int], dtype: MLMultiArrayDataType) throws -> MLMultiArray {
    try MLMultiArray(shape: shape.map(NSNumber.init), dataType: dtype)
}

func zeroFill(_ array: MLMultiArray) {
    let count = array.count
    switch array.dataType {
    case .float16:
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { ptr[i] = 0 }
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { ptr[i] = 0 }
    default:
        break
    }
}

func copyFrameDataIntoArray(
    array: MLMultiArray,
    values: [Float],
    validFrames: Int,
    totalFramesInArray: Int,
    height: Int,
    width: Int,
    channels: Int
) {
    let frameElems = height * width * channels
    let totalElems = totalFramesInArray * frameElems
    switch array.dataType {
    case .float16:
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: totalElems)
        for i in 0..<totalElems { ptr[i] = 0 }
        for i in 0..<(validFrames * frameElems) {
            ptr[i] = Float16(values[i])
        }
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: totalElems)
        for i in 0..<totalElems { ptr[i] = 0 }
        for i in 0..<(validFrames * frameElems) {
            ptr[i] = values[i]
        }
    default:
        break
    }
}

func generateLatentValues(frames: Int, height: Int, width: Int, channels: Int, seed: UInt64) -> [Float] {
    var rng = LCG(seed: seed)
    let count = frames * height * width * channels
    var out = [Float](repeating: 0, count: count)
    for i in 0..<count {
        out[i] = rng.nextFloat(min: -1.0, max: 1.0)
    }
    return out
}

func sliceLatentFrames(
    latent: [Float],
    startFrame: Int,
    frameCount: Int,
    height: Int,
    width: Int,
    channels: Int
) -> [Float] {
    let frameElems = height * width * channels
    let start = startFrame * frameElems
    let end = start + frameCount * frameElems
    return Array(latent[start..<end])
}

func compareFrameRanges(
    chunkArray: MLMultiArray,
    chunkFrameOffset: Int,
    referenceArray: MLMultiArray,
    referenceFrameOffset: Int,
    frameCount: Int,
    frameElems: Int,
    stats: inout DiffStats
) {
    let n = frameCount * frameElems
    switch (chunkArray.dataType, referenceArray.dataType) {
    case (.float16, .float16):
        let a = chunkArray.dataPointer.bindMemory(to: Float16.self, capacity: chunkArray.count)
        let b = referenceArray.dataPointer.bindMemory(to: Float16.self, capacity: referenceArray.count)
        let aBase = chunkFrameOffset * frameElems
        let bBase = referenceFrameOffset * frameElems
        for i in 0..<n {
            stats.add(diff: Float(a[aBase + i]) - Float(b[bBase + i]))
        }
    case (.float16, .float32):
        let a = chunkArray.dataPointer.bindMemory(to: Float16.self, capacity: chunkArray.count)
        let b = referenceArray.dataPointer.bindMemory(to: Float.self, capacity: referenceArray.count)
        let aBase = chunkFrameOffset * frameElems
        let bBase = referenceFrameOffset * frameElems
        for i in 0..<n {
            stats.add(diff: Float(a[aBase + i]) - b[bBase + i])
        }
    case (.float32, .float16):
        let a = chunkArray.dataPointer.bindMemory(to: Float.self, capacity: chunkArray.count)
        let b = referenceArray.dataPointer.bindMemory(to: Float16.self, capacity: referenceArray.count)
        let aBase = chunkFrameOffset * frameElems
        let bBase = referenceFrameOffset * frameElems
        for i in 0..<n {
            stats.add(diff: a[aBase + i] - Float(b[bBase + i]))
        }
    case (.float32, .float32):
        let a = chunkArray.dataPointer.bindMemory(to: Float.self, capacity: chunkArray.count)
        let b = referenceArray.dataPointer.bindMemory(to: Float.self, capacity: referenceArray.count)
        let aBase = chunkFrameOffset * frameElems
        let bBase = referenceFrameOffset * frameElems
        for i in 0..<n {
            stats.add(diff: a[aBase + i] - b[bBase + i])
        }
    default:
        break
    }
}

func runStatefulDecode(
    statefulModel: MLModel,
    latentValues: [Float],
    latentFrames: Int,
    chunkT: Int,
    latentHW: Int,
    latentChannels: Int,
    trimFrames: Int,
    dtype: MLMultiArrayDataType,
    actInputNames: [String],
    actOutputNames: [String],
    actInputShapes: [[Int]],
    tUpscale: Int,
    outputFrameElems: Int,
    referenceOutput: MLMultiArray?,
    referenceWantedFrames: Int
) throws -> DiffStats {
    var states = try actInputShapes.map { try makeMultiArray(shape: $0, dtype: dtype) }
    for s in states { zeroFill(s) }

    var producedRawFrames = 0
    var diffStats = DiffStats()

    for start in stride(from: 0, to: latentFrames, by: chunkT) {
        let realT = min(chunkT, latentFrames - start)
        let chunkLatent = try makeMultiArray(shape: [1, chunkT, latentHW, latentHW, latentChannels], dtype: dtype)
        let values = sliceLatentFrames(
            latent: latentValues,
            startFrame: start,
            frameCount: realT,
            height: latentHW,
            width: latentHW,
            channels: latentChannels
        )
        copyFrameDataIntoArray(
            array: chunkLatent,
            values: values,
            validFrames: realT,
            totalFramesInArray: chunkT,
            height: latentHW,
            width: latentHW,
            channels: latentChannels
        )

        var features: [String: MLFeatureValue] = ["latent": MLFeatureValue(multiArray: chunkLatent)]
        for (idx, stateName) in actInputNames.enumerated() {
            features[stateName] = MLFeatureValue(multiArray: states[idx])
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let out = try statefulModel.prediction(from: provider)
        guard let outImage = out.featureValue(for: "image")?.multiArrayValue else {
            throw ArgError.message("Stateful model output 'image' not found")
        }

        if let ref = referenceOutput {
            let rawFramesInChunk = realT * tUpscale
            for localRawFrame in 0..<rawFramesInChunk {
                let globalRawFrame = producedRawFrames + localRawFrame
                let outputFrame = globalRawFrame - trimFrames
                if outputFrame < 0 { continue }
                if outputFrame >= referenceWantedFrames { break }
                compareFrameRanges(
                    chunkArray: outImage,
                    chunkFrameOffset: localRawFrame,
                    referenceArray: ref,
                    referenceFrameOffset: outputFrame,
                    frameCount: 1,
                    frameElems: outputFrameElems,
                    stats: &diffStats
                )
            }
        }

        states = actOutputNames.map { out.featureValue(for: $0)!.multiArrayValue! }
        producedRawFrames += realT * tUpscale
    }

    return diffStats
}

func run() throws {
    let cfg = try parseArgs()

    let modelConfig = MLModelConfiguration()
    modelConfig.computeUnits = cfg.computeUnits

    let statefulModel = try MLModel(
        contentsOf: URL(fileURLWithPath: cfg.statefulModelPath),
        configuration: modelConfig
    )

    guard let latentIn = statefulModel.modelDescription.inputDescriptionsByName["latent"]?.multiArrayConstraint else {
        throw ArgError.message("Stateful input 'latent' not found")
    }
    guard let imageOut = statefulModel.modelDescription.outputDescriptionsByName["image"]?.multiArrayConstraint else {
        throw ArgError.message("Stateful output 'image' not found")
    }

    let latentShape = shapeFromConstraint(latentIn)
    if latentShape.count != 5 {
        throw ArgError.message("Expected stateful latent input rank 5, got shape \(latentShape)")
    }
    let chunkT = latentShape[1]
    let latentHW = latentShape[2]
    let latentChannels = latentShape[4]
    let imageShape = shapeFromConstraint(imageOut)
    if imageShape.count != 5 {
        throw ArgError.message("Expected stateful image output rank 5, got shape \(imageShape)")
    }
    let rawFramesPerCall = imageShape[1]
    if chunkT <= 0 || rawFramesPerCall % chunkT != 0 {
        throw ArgError.message("Invalid model temporal factors. chunkT=\(chunkT), outputT=\(rawFramesPerCall)")
    }
    let tUpscale = rawFramesPerCall / chunkT
    let outputH = imageShape[2]
    let outputW = imageShape[3]
    let outputC = imageShape[4]
    let outputFrameElems = outputH * outputW * outputC

    let inferredType = latentIn.dataType
    let dtype = cfg.dtypeOverride ?? inferredType
    if dtype != .float16 && dtype != .float32 {
        throw ArgError.message("Only float16/float32 are supported, got \(dtype.rawValue)")
    }

    let actInputNames = sortedActNames(Array(statefulModel.modelDescription.inputDescriptionsByName.keys), isOutput: false)
    let actOutputNames = sortedActNames(Array(statefulModel.modelDescription.outputDescriptionsByName.keys), isOutput: true)
    if actInputNames.isEmpty || actInputNames.count != actOutputNames.count {
        throw ArgError.message("Could not resolve activation input/output names.")
    }
    let actInputShapes = try actInputNames.map { name -> [Int] in
        guard let c = statefulModel.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint else {
            throw ArgError.message("Missing state input '\(name)'")
        }
        return shapeFromConstraint(c)
    }

    var referenceModel: MLModel?
    var referenceT = 0
    if let refPath = cfg.referenceModelPath {
        let ref = try MLModel(contentsOf: URL(fileURLWithPath: refPath), configuration: modelConfig)
        guard let refLatent = ref.modelDescription.inputDescriptionsByName["latent"]?.multiArrayConstraint else {
            throw ArgError.message("Reference model input 'latent' not found")
        }
        let refLatentShape = shapeFromConstraint(refLatent)
        if refLatentShape.count != 5 || refLatentShape[2] != latentHW || refLatentShape[4] != latentChannels {
            throw ArgError.message("Reference latent shape \(refLatentShape) is incompatible with stateful latent shape \(latentShape).")
        }
        referenceT = refLatentShape[1]
        referenceModel = ref
    }

    print("Stateful model: \(cfg.statefulModelPath)")
    if let refPath = cfg.referenceModelPath {
        print("Reference model: \(refPath)")
    }
    print("ComputeUnits: \(cfg.computeUnits.rawValue)")
    print("Stateful latent shape: \(latentShape)")
    print("Stateful image shape per call: \(imageShape)")
    print("Activation tensors: \(actInputNames.count)")
    print("Temporal: chunkT=\(chunkT), tUpscale=\(tUpscale), trim=\(cfg.trimFrames)")
    print("Input dtype: \(dtype == .float16 ? "float16" : "float32")")

    for (idx, t) in cfg.lengths.enumerated() {
        if t <= 0 { throw ArgError.message("All --lengths values must be positive, got \(t)") }
        var latentSeed = cfg.seed &+ UInt64(idx * 9973)
        if latentSeed == 0 { latentSeed = UInt64(t * 31 + 7) }
        let latent = generateLatentValues(
            frames: t,
            height: latentHW,
            width: latentHW,
            channels: latentChannels,
            seed: latentSeed
        )

        if cfg.checkCorrectness {
            guard let refModel = referenceModel else {
                throw ArgError.message("Correctness check requested without reference model.")
            }
            if t > referenceT {
                throw ArgError.message("Length \(t) exceeds reference model latent T=\(referenceT).")
            }
            let refLatentArray = try makeMultiArray(shape: [1, referenceT, latentHW, latentHW, latentChannels], dtype: dtype)
            copyFrameDataIntoArray(
                array: refLatentArray,
                values: latent,
                validFrames: t,
                totalFramesInArray: referenceT,
                height: latentHW,
                width: latentHW,
                channels: latentChannels
            )
            let refOut = try refModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: refLatentArray)
            ]))
            guard let refImage = refOut.featureValue(for: "image")?.multiArrayValue else {
                throw ArgError.message("Reference output 'image' not found.")
            }
            let expectedFrames = t * tUpscale - cfg.trimFrames
            let diff = try runStatefulDecode(
                statefulModel: statefulModel,
                latentValues: latent,
                latentFrames: t,
                chunkT: chunkT,
                latentHW: latentHW,
                latentChannels: latentChannels,
                trimFrames: cfg.trimFrames,
                dtype: dtype,
                actInputNames: actInputNames,
                actOutputNames: actOutputNames,
                actInputShapes: actInputShapes,
                tUpscale: tUpscale,
                outputFrameElems: outputFrameElems,
                referenceOutput: refImage,
                referenceWantedFrames: expectedFrames
            )
            print(String(format: "Correctness T=%d: MAE=%.6g MaxAbs=%.6g", t, diff.mae, diff.maxAbs))
        }

        for _ in 0..<cfg.warmup {
            _ = try runStatefulDecode(
                statefulModel: statefulModel,
                latentValues: latent,
                latentFrames: t,
                chunkT: chunkT,
                latentHW: latentHW,
                latentChannels: latentChannels,
                trimFrames: cfg.trimFrames,
                dtype: dtype,
                actInputNames: actInputNames,
                actOutputNames: actOutputNames,
                actInputShapes: actInputShapes,
                tUpscale: tUpscale,
                outputFrameElems: outputFrameElems,
                referenceOutput: nil,
                referenceWantedFrames: 0
            )
        }

        var timingsMs: [Double] = []
        timingsMs.reserveCapacity(cfg.iterations)
        for _ in 0..<cfg.iterations {
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try runStatefulDecode(
                statefulModel: statefulModel,
                latentValues: latent,
                latentFrames: t,
                chunkT: chunkT,
                latentHW: latentHW,
                latentChannels: latentChannels,
                trimFrames: cfg.trimFrames,
                dtype: dtype,
                actInputNames: actInputNames,
                actOutputNames: actOutputNames,
                actInputShapes: actInputShapes,
                tUpscale: tUpscale,
                outputFrameElems: outputFrameElems,
                referenceOutput: nil,
                referenceWantedFrames: 0
            )
            let t1 = DispatchTime.now().uptimeNanoseconds
            timingsMs.append(Double(t1 - t0) / 1_000_000.0)
        }

        let mean = timingsMs.reduce(0, +) / Double(timingsMs.count)
        let median = percentile(timingsMs, 0.5)
        let p90 = percentile(timingsMs, 0.9)
        let p95 = percentile(timingsMs, 0.95)
        let minV = timingsMs.min() ?? 0
        let maxV = timingsMs.max() ?? 0
        let outFrames = max(0, t * tUpscale - cfg.trimFrames)
        let msPerFrame = outFrames > 0 ? mean / Double(outFrames) : 0
        let fps = outFrames > 0 ? (1000.0 * Double(outFrames) / mean) : 0

        print(String(format: "Perf T=%d: mean=%.3f ms median=%.3f p90=%.3f p95=%.3f min=%.3f max=%.3f", t, mean, median, p90, p95, minV, maxV))
        print(String(format: "Perf T=%d: output_frames=%d ms_per_frame=%.3f fps=%.2f", t, outFrames, msPerFrame, fps))
    }
}

do {
    try run()
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
