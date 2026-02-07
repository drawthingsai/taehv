import Foundation
import CoreML

struct Shape {
    let dims: [Int]
    var count: Int { dims.reduce(1, *) }
}

func readShape(_ path: String) throws -> Shape {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let parts = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" })
    let dims = parts.compactMap { Int($0) }
    return Shape(dims: dims)
}

func readRaw(_ path: String) throws -> Data {
    return try Data(contentsOf: URL(fileURLWithPath: path))
}

func loadFloat16Array(_ data: Data) -> [Float16] {
    let count = data.count / MemoryLayout<UInt16>.size
    var out = [Float16](repeating: 0, count: count)
    data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
        let u16 = ptr.bindMemory(to: UInt16.self)
        for i in 0..<count {
            out[i] = Float16(bitPattern: u16[i])
        }
    }
    return out
}

func loadFloat32Array(_ data: Data) -> [Float] {
    let count = data.count / MemoryLayout<Float>.size
    return data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
        let f = ptr.bindMemory(to: Float.self)
        return Array(f.prefix(count))
    }
}

func writeToMLMultiArrayFloat16(_ array: MLMultiArray, values: [Float16]) {
    let count = values.count
    let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
    for i in 0..<count {
        ptr[i] = values[i]
    }
}

func writeToMLMultiArrayFloat32(_ array: MLMultiArray, values: [Float]) {
    let count = values.count
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    for i in 0..<count {
        ptr[i] = values[i]
    }
}

func readMLMultiArrayFloat16(_ array: MLMultiArray) -> [Float] {
    let count = array.count
    let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
    var out = [Float](repeating: 0, count: count)
    for i in 0..<count {
        out[i] = Float(ptr[i])
    }
    return out
}

func readMLMultiArrayFloat32(_ array: MLMultiArray) -> [Float] {
    let count = array.count
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    var out = [Float](repeating: 0, count: count)
    for i in 0..<count {
        out[i] = ptr[i]
    }
    return out
}

func computeDiffStats(_ a: [Float], _ b: [Float]) -> (mae: Float, maxAbs: Float) {
    precondition(a.count == b.count)
    var sumAbs: Float = 0
    var maxAbs: Float = 0
    for i in 0..<a.count {
        let d = abs(a[i] - b[i])
        sumAbs += d
        if d > maxAbs { maxAbs = d }
    }
    return (sumAbs / Float(a.count), maxAbs)
}

let args = CommandLine.arguments
if args.count < 6 {
    print("Usage: validate_wan21_coreml.swift <model.mlmodelc> <input.raw> <output_expected.raw> <input_shape.txt> <output_shape.txt> [float16|float32]")
    exit(1)
}

let modelPath = args[1]
let inputPath = args[2]
let expectedPath = args[3]
let inputShapePath = args[4]
let outputShapePath = args[5]
let dtype = args.count > 6 ? args[6] : "float16"

let inputShape = try readShape(inputShapePath)
let outputShape = try readShape(outputShapePath)

let inputData = try readRaw(inputPath)
let expectedData = try readRaw(expectedPath)

let config = MLModelConfiguration()
let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: config)

let inputArray = try MLMultiArray(shape: inputShape.dims as [NSNumber], dataType: dtype == "float32" ? .float32 : .float16)
if dtype == "float32" {
    let values = loadFloat32Array(inputData)
    writeToMLMultiArrayFloat32(inputArray, values: values)
} else {
    let values = loadFloat16Array(inputData)
    writeToMLMultiArrayFloat16(inputArray, values: values)
}

let provider = try MLDictionaryFeatureProvider(dictionary: ["latent": inputArray])
let output = try model.prediction(from: provider)
let outputArray = output.featureValue(for: "image")!.multiArrayValue!

let expected: [Float]
if dtype == "float32" {
    expected = loadFloat32Array(expectedData)
} else {
    expected = loadFloat16Array(expectedData).map { Float($0) }
}

let actual: [Float]
if outputArray.dataType == .float32 {
    actual = readMLMultiArrayFloat32(outputArray)
} else {
    actual = readMLMultiArrayFloat16(outputArray)
}

let stats = computeDiffStats(actual, expected)
print(String(format: "MAE: %.6g", stats.mae))
print(String(format: "Max abs: %.6g", stats.maxAbs))
