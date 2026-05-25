// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "yolo26-coreml",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "yolo26-coreml",
            path: "Sources"
        )
    ]
)
