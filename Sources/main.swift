/// yolo26-coreml: Minimal CoreML inference CLI for YOLO26
///
/// Usage:
///   yolo26-coreml predict <model.mlpackage> <image.jpg> [--compute-units ane|all|gpu|cpu]
///   yolo26-coreml bench <model.mlpackage> <image.jpg> [--runs 100] [--warmup 3] [--compute-units ane]
///
/// Output: JSON array of detections to stdout
///   [{"x": f32, "y": f32, "w": f32, "h": f32, "conf": f32, "class": i32}, ...]
///
/// Designed to be called directly or from the Rust wrapper via subprocess.

import CoreML
import Foundation
import CoreImage
import AppKit
import AVFoundation
import CoreMedia
import CoreVideo

// MARK: - Types

struct Detection: Codable {
    let x: Float
    let y: Float
    let w: Float
    let h: Float
    let conf: Float
    let classId: Int

    enum CodingKeys: String, CodingKey {
        case x, y, w, h, conf
        case classId = "class"
    }
}

struct BenchResult: Codable {
    let fps: Float
    let meanMs: Float
    let stdMs: Float
    let medianMs: Float
    let p50Ms: Float
    let p99Ms: Float
    let minMs: Float
    let maxMs: Float
    let warmup: Int
    let runs: Int
    let computeUnits: String
    let cacheBust: Bool
    let preprocess: String

    enum CodingKeys: String, CodingKey {
        case fps
        case meanMs = "mean_ms"
        case stdMs = "std_ms"
        case medianMs = "median_ms"
        case p50Ms = "p50_ms"
        case p99Ms = "p99_ms"
        case minMs = "min_ms"
        case maxMs = "max_ms"
        case warmup
        case runs
        case computeUnits = "compute_units"
        case cacheBust = "cache_bust"
        case preprocess
    }
}

struct DetectionFrame: Codable {
    let timestampMs: Int
    let frame: Int
    let detections: [Detection]

    enum CodingKeys: String, CodingKey {
        case timestampMs = "timestamp_ms"
        case frame
        case detections
    }
}

struct ImagePredictionFrame: Codable {
    let image: String
    let elapsedMs: Float
    let detections: [Detection]

    enum CodingKeys: String, CodingKey {
        case image
        case elapsedMs = "elapsed_ms"
        case detections
    }
}

// MARK: - CoreML Loading

func loadModel(path: String, computeUnits: MLComputeUnits) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits

    // .mlpackage needs compilation first; .mlmodelc can be loaded directly
    if path.hasSuffix(".mlpackage") {
        fputs("Compiling .mlpackage to .mlmodelc...\n", stderr)
        let compiledURL = try MLModel.compileModel(at: url)
        fputs("Compiled to: \(compiledURL.path)\n", stderr)
        return try MLModel(contentsOf: compiledURL, configuration: config)
    } else {
        return try MLModel(contentsOf: url, configuration: config)
    }
}

func parseComputeUnits(_ str: String) -> MLComputeUnits {
    switch str.lowercased() {
    case "ane", "ne", "cpu_and_ne":
        return .cpuAndNeuralEngine
    case "gpu", "cpu_and_gpu":
        return .cpuAndGPU
    case "cpu":
        return .cpuOnly
    default:
        return .all
    }
}

// MARK: - Image Loading

func loadImage(path: String, size: Int = 640, preprocess: String = "stretch") throws -> CVPixelBuffer {
    guard let image = NSImage(contentsOfFile: path) else {
        throw NSError(domain: "yolo26", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot load image: \(path)"])
    }

    // Create pixel buffer
    var pixelBuffer: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferWidthKey: size,
        kCVPixelBufferHeightKey: size,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        size, size,
        kCVPixelFormatType_32BGRA,
        attrs as CFDictionary,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        throw NSError(domain: "yolo26", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot create pixel buffer"])
    }

    // Draw image into pixel buffer
    CVPixelBufferLockBaseAddress(buffer, [])
    let context = CGContext(
        data: CVPixelBufferGetBaseAddress(buffer),
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!

    let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)!

    if preprocess.lowercased() == "letterbox" {
        context.setFillColor(CGColor(red: 114.0 / 255.0, green: 114.0 / 255.0, blue: 114.0 / 255.0, alpha: 1.0))
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        let srcWidth = CGFloat(cgImage.width)
        let srcHeight = CGFloat(cgImage.height)
        let scale = min(CGFloat(size) / srcWidth, CGFloat(size) / srcHeight)
        let destWidth = srcWidth * scale
        let destHeight = srcHeight * scale
        let destX = (CGFloat(size) - destWidth) / 2.0
        let destY = (CGFloat(size) - destHeight) / 2.0
        context.draw(cgImage, in: CGRect(x: destX, y: destY, width: destWidth, height: destHeight))
    } else {
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
    }

    CVPixelBufferUnlockBaseAddress(buffer, [])

    return buffer
}

// MARK: - Inference

func predict(model: MLModel, pixelBuffer: CVPixelBuffer) throws -> [Detection] {
    // CoreML model from Ultralytics expects input named "image"
    let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
    let output = try model.prediction(from: input)

    // Output is a MLMultiArray with shape (1, 300, 6): [x, y, w, h, conf, class]
    guard let multiArray = output.featureValue(for: "var_1441")?.multiArrayValue else {
        // Try other common output names
        let names = output.featureNames
        for name in names {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return parseDetections(arr)
            }
        }
        throw NSError(domain: "yolo26", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "No MLMultiArray output found. Available: \(output.featureNames)"
        ])
    }

    return parseDetections(multiArray)
}

func parseDetections(_ multiArray: MLMultiArray) -> [Detection] {
    // Shape: (1, 300, 6) or (300, 6)
    let shape = multiArray.shape.map { $0.intValue }
    let numDets: Int
    let stride: Int

    if shape.count == 3 {
        numDets = shape[1]
        stride = shape[2]
    } else if shape.count == 2 {
        numDets = shape[0]
        stride = shape[1]
    } else {
        return []
    }

    var detections: [Detection] = []
    let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)

    for i in 0..<numDets {
        let offset = (shape.count == 3 ? shape[1] * shape[2] * 0 : 0) + i * stride
        let conf = ptr[offset + 4]
        if conf.isFinite && conf >= 0 {
            detections.append(Detection(
                x: ptr[offset + 0],
                y: ptr[offset + 1],
                w: ptr[offset + 2],
                h: ptr[offset + 3],
                conf: conf,
                classId: Int(ptr[offset + 5])
            ))
        }
    }

    return detections
}

// MARK: - Benchmark

func perturbPixelBuffer(_ pixelBuffer: CVPixelBuffer, iteration: Int) {
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let pixelIndex = (iteration * 9973) % max(width * height, 1)
    let x = pixelIndex % width
    let y = pixelIndex / width
    let ptr = base.assumingMemoryBound(to: UInt8.self)
    let offset = y * bytesPerRow + x * 4

    ptr[offset + 0] = UInt8((iteration * 17) & 0xff)
    ptr[offset + 1] = UInt8((iteration * 31) & 0xff)
    ptr[offset + 2] = UInt8((iteration * 47) & 0xff)
    ptr[offset + 3] = 255
}

func benchmark(model: MLModel, pixelBuffer: CVPixelBuffer, runs: Int, warmup: Int, computeUnitsLabel: String, cacheBust: Bool, preprocess: String) throws -> BenchResult {
    let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])

    // Warmup
    for i in 0..<warmup {
        if cacheBust {
            perturbPixelBuffer(pixelBuffer, iteration: i)
        }
        _ = try model.prediction(from: input)
    }

    // Timed runs
    var latencies: [Double] = []
    for i in 0..<runs {
        if cacheBust {
            perturbPixelBuffer(pixelBuffer, iteration: i + warmup)
        }
        let start = CFAbsoluteTimeGetCurrent()
        _ = try model.prediction(from: input)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        latencies.append(elapsed)
    }

    latencies.sort()
    let mean = latencies.reduce(0, +) / Double(runs)
    let variance = latencies.reduce(0.0) { acc, value in
        let diff = value - mean
        return acc + diff * diff
    } / Double(runs)
    let p50 = latencies[runs / 2]
    let p99Index = min(runs - 1, Int(Double(runs) * 0.99))
    let p99 = latencies[p99Index]

    return BenchResult(
        fps: Float(1000.0 / mean),
        meanMs: Float(mean),
        stdMs: Float(sqrt(variance)),
        medianMs: Float(p50),
        p50Ms: Float(p50),
        p99Ms: Float(p99),
        minMs: Float(latencies.first!),
        maxMs: Float(latencies.last!),
        warmup: warmup,
        runs: runs,
        computeUnits: computeUnitsLabel,
        cacheBust: cacheBust,
        preprocess: preprocess
    )
}

func runPredictList(model: MLModel, listPath: String, confThreshold: Float, preprocess: String) throws {
    let listURL = URL(fileURLWithPath: listPath)
    let contents = try String(contentsOf: listURL, encoding: .utf8)
    let paths = contents
        .split(whereSeparator: \.isNewline)
        .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    let encoder = JSONEncoder()

    for path in paths {
        let pixelBuffer = try loadImage(path: path, preprocess: preprocess)
        let start = CFAbsoluteTimeGetCurrent()
        let detections = try predict(model: model, pixelBuffer: pixelBuffer)
            .filter { $0.conf >= confThreshold }
        let elapsed = Float((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
        let frame = ImagePredictionFrame(image: path, elapsedMs: elapsed, detections: detections)
        let json = try encoder.encode(frame)
        print(String(data: json, encoding: .utf8)!)
        fflush(stdout)
    }
}

// MARK: - Main

func main() throws {
    let args = CommandLine.arguments

    guard args.count >= 3 else {
        fputs("""
        Usage:
          yolo26-coreml predict <model.mlpackage> <image.jpg> [--compute-units ane|all|gpu|cpu] [--conf 0.25]
          yolo26-coreml predict-list <model.mlpackage> <images.txt> [--compute-units ane|all|gpu|cpu] [--conf 0.001] [--preprocess stretch|letterbox]
          yolo26-coreml bench <model.mlpackage> <image.jpg> [--runs 100] [--warmup 3] [--compute-units ane] [--cache-bust] [--preprocess stretch|letterbox]
          yolo26-coreml webcam <model.mlpackage> [--compute-units ane] [--conf 0.25]
          yolo26-coreml webcam-json <model.mlpackage> [--compute-units ane] [--conf 0.25] [--max-frames 0]

        """, stderr)
        exit(1)
    }

    let command = args[1]
    let modelPath = args[2]
    let imagePath = args.count > 3 ? args[3] : ""
    let commandHasImage = command == "predict" || command == "bench" || command == "predict-list"

    // Parse optional args
    var computeUnitsStr = "ane"
    var runs = 100
    var warmup = 20
    var confThreshold: Float = 0.25
    var maxFrames = 0
    var cacheBust = false
    var preprocess = "stretch"

    var i = commandHasImage ? 4 : 3
    while i < args.count {
        switch args[i] {
        case "--compute-units":
            i += 1; computeUnitsStr = args[i]
        case "--runs":
            i += 1; runs = Int(args[i]) ?? 100
        case "--warmup":
            i += 1; warmup = Int(args[i]) ?? warmup
        case "--conf":
            i += 1; confThreshold = Float(args[i]) ?? 0.25
        case "--max-frames":
            i += 1; maxFrames = Int(args[i]) ?? 0
        case "--cache-bust":
            cacheBust = true
        case "--preprocess":
            i += 1; preprocess = args[i]
        default:
            break
        }
        i += 1
    }

    let computeUnits = parseComputeUnits(computeUnitsStr)

    // Load model
    fputs("Loading model: \(modelPath) (compute_units: \(computeUnitsStr))\n", stderr)
    let model = try loadModel(path: modelPath, computeUnits: computeUnits)

    // Webcam doesn't need an image
    if command == "webcam" {
        try runWebcam(model: model, confThreshold: confThreshold)
        return
    }
    if command == "webcam-json" {
        try runWebcamJson(model: model, confThreshold: confThreshold, maxFrames: maxFrames)
        return
    }
    if command == "predict-list" {
        try runPredictList(
            model: model,
            listPath: imagePath,
            confThreshold: confThreshold,
            preprocess: preprocess
        )
        return
    }

    // Load image (required for predict/bench)
    fputs("Loading image: \(imagePath)\n", stderr)
    let pixelBuffer = try loadImage(path: imagePath, preprocess: preprocess)

    switch command {
    case "predict":
        let detections = try predict(model: model, pixelBuffer: pixelBuffer)
        let filtered = detections.filter { $0.conf >= confThreshold }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let json = try encoder.encode(filtered)
        print(String(data: json, encoding: .utf8)!)
        fputs("Detections: \(filtered.count) (conf >= \(confThreshold))\n", stderr)

    case "bench":
        let result = try benchmark(
            model: model,
            pixelBuffer: pixelBuffer,
            runs: runs,
            warmup: warmup,
            computeUnitsLabel: computeUnitsStr,
            cacheBust: cacheBust,
            preprocess: preprocess
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let json = try encoder.encode(result)
        print(String(data: json, encoding: .utf8)!)
        fputs("Benchmark: \(result.fps) FPS, \(result.p50Ms)ms p50 (\(runs) runs, cache_bust=\(cacheBust))\n", stderr)

    default:
        fputs("Unknown command: \(command)\n", stderr)
        exit(1)
    }
}

// MARK: - Webcam Live Detection

func runWebcam(model: MLModel, confThreshold: Float) throws {
    // Initialize NSApplication for window display
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)

    fputs("Starting webcam detection with visual overlay...\n", stderr)
    fputs("Press Ctrl+C to stop.\n", stderr)

    let session = AVCaptureSession()
    session.sessionPreset = .hd1280x720

    guard let device = AVCaptureDevice.default(for: .video) else {
        fputs("ERROR: No camera found\n", stderr)
        exit(1)
    }

    let input = try AVCaptureDeviceInput(device: device)
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]

    // Process frames on a serial queue
    let queue = DispatchQueue(label: "yolo26.camera")
    let delegate = CameraDelegate(model: model, confThreshold: confThreshold, showWindow: true)
    output.setSampleBufferDelegate(delegate, queue: queue)
    output.alwaysDiscardsLateVideoFrames = true
    session.addOutput(output)

    session.startRunning()
    fputs("Camera started. Running inference on ANE. Window should appear.\n", stderr)

    // Run until Ctrl+C
    signal(SIGINT) { _ in
        fputs("\nStopping...\n", stderr)
        exit(0)
    }

    app.activate(ignoringOtherApps: true)
    app.run()
}

func runWebcamJson(model: MLModel, confThreshold: Float, maxFrames: Int) throws {
    fputs("Starting webcam JSON detection stream...\n", stderr)
    fputs("Writing one JSON frame per line to stdout. Press Ctrl+C to stop.\n", stderr)

    let session = AVCaptureSession()
    session.sessionPreset = .hd1280x720

    guard let device = AVCaptureDevice.default(for: .video) else {
        fputs("ERROR: No camera found\n", stderr)
        exit(1)
    }

    let input = try AVCaptureDeviceInput(device: device)
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]

    let queue = DispatchQueue(label: "yolo26.camera.json")
    let delegate = JsonCameraDelegate(model: model, confThreshold: confThreshold, maxFrames: maxFrames)
    output.setSampleBufferDelegate(delegate, queue: queue)
    output.alwaysDiscardsLateVideoFrames = true
    session.addOutput(output)

    signal(SIGINT) { _ in
        fputs("\nStopping...\n", stderr)
        exit(0)
    }

    session.startRunning()
    RunLoop.current.run()
}

// COCO class names for display
let cocoClasses = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class CameraDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let model: MLModel
    let confThreshold: Float
    var frameCount = 0
    var lastPrintTime = CFAbsoluteTimeGetCurrent()
    var fpsAccumulator: [Double] = []
    let ciContext = CIContext(options: [.useSoftwareRenderer: false])  // GPU-accelerated resize
    var lastDetections: [Detection] = []
    var lastFps: Double = 0

    // Overlay window
    var overlayWindow: NSWindow?
    var imageView: NSImageView?
    var lastDisplayTime: CFAbsoluteTime = 0
    let displayInterval: CFAbsoluteTime = 1.0 / 30.0  // Cap display at 30 FPS

    init(model: MLModel, confThreshold: Float, showWindow: Bool = true) {
        self.model = model
        self.confThreshold = confThreshold
        super.init()

        if showWindow {
            DispatchQueue.main.async { [weak self] in
                self?.setupWindow()
            }
        }
    }

    func setupWindow() {
        let window = NSWindow(
            contentRect: NSRect(x: 100, y: 100, width: 1280, height: 720),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "YOLO26 ANE — Live Detection"
        window.isReleasedWhenClosed = false

        let view = NSImageView(frame: window.contentView!.bounds)
        view.autoresizingMask = [.width, .height]
        view.imageScaling = .scaleProportionallyUpOrDown
        window.contentView?.addSubview(view)

        window.makeKeyAndOrderFront(nil)
        self.overlayWindow = window
        self.imageView = view
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Resize to 640x640 for model input
        let resized = resizePixelBuffer(pixelBuffer, width: 640, height: 640)
        guard let input = resized else { return }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Run inference
        do {
            let featureProvider = try MLDictionaryFeatureProvider(dictionary: ["image": input])
            let result = try model.prediction(from: featureProvider)

            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            let fps = 1000.0 / elapsed

            // Parse detections
            var detections: [Detection] = []
            for name in result.featureNames {
                if let arr = result.featureValue(for: name)?.multiArrayValue {
                    detections = parseDetections(arr).filter { $0.conf >= confThreshold }
                    break
                }
            }

            frameCount += 1
            fpsAccumulator.append(fps)
            lastDetections = detections
            lastFps = fps

            // Throttle display to 30 FPS (inference runs faster but display can't keep up)
            let now2 = CFAbsoluteTimeGetCurrent()
            if now2 - lastDisplayTime >= displayInterval {
                lastDisplayTime = now2
                let frameImage = CIImage(cvPixelBuffer: pixelBuffer)
                let capturedDets = detections
                let capturedFps = fps
                DispatchQueue.main.async { [weak self] in
                    self?.updateDisplay(frame: frameImage, detections: capturedDets, fps: capturedFps)
                }
            }

            // Print stats every second
            let now = CFAbsoluteTimeGetCurrent()
            if now - lastPrintTime >= 1.0 {
                let avgFps = fpsAccumulator.reduce(0, +) / Double(fpsAccumulator.count)
                let classes = detections.map { d in
                    d.classId < cocoClasses.count ? cocoClasses[d.classId] : "?\(d.classId)"
                }.joined(separator: ", ")
                fputs("\r\u{1B}[K[ANE] \(String(format: "%.1f", avgFps)) FPS | \(String(format: "%.2f", elapsed))ms | \(detections.count) det [\(classes)] | #\(frameCount)", stderr)
                fpsAccumulator = []
                lastPrintTime = now
            }
        } catch {
            fputs("Inference error: \(error)\n", stderr)
        }
    }

    func updateDisplay(frame: CIImage, detections: [Detection], fps: Double) {
        guard let imageView = imageView else { return }

        // Render frame with bounding box overlay
        let frameWidth = frame.extent.width
        let frameHeight = frame.extent.height

        let rep = NSCIImageRep(ciImage: frame)
        let nsImage = NSImage(size: NSSize(width: frameWidth, height: frameHeight))
        nsImage.addRepresentation(rep)

        // Draw boxes on top
        nsImage.lockFocus()
        let ctx = NSGraphicsContext.current!.cgContext

        // Scale detections from 640x640 model space to frame space
        let scaleX = frameWidth / 640.0
        let scaleY = frameHeight / 640.0

        for det in detections {
            let x = CGFloat(det.x) * scaleX
            let y = CGFloat(det.y) * scaleY
            let w = CGFloat(det.w - det.x) * scaleX  // w is x2, not width
            let h = CGFloat(det.h - det.y) * scaleY  // h is y2, not height

            // CIImage is flipped, adjust y
            let flippedY = frameHeight - y - h

            // Draw box
            ctx.setStrokeColor(NSColor.green.cgColor)
            ctx.setLineWidth(3.0)
            ctx.stroke(CGRect(x: x, y: flippedY, width: w, height: h))

            // Draw label
            let className = det.classId < cocoClasses.count ? cocoClasses[det.classId] : "?"
            let label = "\(className) \(String(format: "%.0f%%", det.conf * 100))"
            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.boldSystemFont(ofSize: 14),
                .foregroundColor: NSColor.green,
                .backgroundColor: NSColor.black.withAlphaComponent(0.6),
            ]
            (label as NSString).draw(at: NSPoint(x: x + 2, y: flippedY + h - 18), withAttributes: attrs)
        }

        // Draw FPS overlay
        let fpsLabel = String(format: "ANE: %.0f FPS | %d det", fps, detections.count)
        let fpsAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 16, weight: .bold),
            .foregroundColor: NSColor.yellow,
            .backgroundColor: NSColor.black.withAlphaComponent(0.7),
        ]
        (fpsLabel as NSString).draw(at: NSPoint(x: 10, y: frameHeight - 25), withAttributes: fpsAttrs)

        nsImage.unlockFocus()
        imageView.image = nsImage
    }

    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resized: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &resized)
        guard let output = resized else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        ciContext.render(scaled, to: output)

        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferUnlockBaseAddress(output, [])

        return output
    }
}

class JsonCameraDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let model: MLModel
    let confThreshold: Float
    let maxFrames: Int
    let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    let encoder = JSONEncoder()
    let startTime = CFAbsoluteTimeGetCurrent()
    var frameCount = 0

    init(model: MLModel, confThreshold: Float, maxFrames: Int) {
        self.model = model
        self.confThreshold = confThreshold
        self.maxFrames = maxFrames
        super.init()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let input = resizePixelBuffer(pixelBuffer, width: 640, height: 640) else { return }

        do {
            let featureProvider = try MLDictionaryFeatureProvider(dictionary: ["image": input])
            let result = try model.prediction(from: featureProvider)
            var detections: [Detection] = []
            for name in result.featureNames {
                if let arr = result.featureValue(for: name)?.multiArrayValue {
                    detections = parseDetections(arr).filter { $0.conf >= confThreshold }
                    break
                }
            }

            let timestampMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000.0)
            let frame = DetectionFrame(timestampMs: timestampMs, frame: frameCount, detections: detections)
            let json = try encoder.encode(frame)
            if let line = String(data: json, encoding: .utf8) {
                print(line)
                fflush(stdout)
            }

            frameCount += 1
            if maxFrames > 0 && frameCount >= maxFrames {
                exit(0)
            }
        } catch {
            fputs("Inference error: \(error)\n", stderr)
        }
    }

    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resized: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &resized)
        guard let output = resized else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        ciContext.render(scaled, to: output)

        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferUnlockBaseAddress(output, [])

        return output
    }
}

try main()
