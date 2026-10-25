// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "haitaka",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Haitaka", targets: ["Haitaka"]),
        .executable(name: "train", targets: ["train"]),
        .executable(name: "transcribe", targets: ["transcribe"]),
        .executable(name: "mictrans", targets: ["mictrans"]),
        .executable(name: "stage2", targets: ["stage2"]),
        .executable(name: "kanalm", targets: ["kanalm"]),
        .executable(name: "kanatext", targets: ["kanatext"]),
        .executable(name: "pretrain", targets: ["pretrain"]),
        .executable(name: "screen", targets: ["screen"]),
        .executable(name: "segment", targets: ["segment"]),
        .executable(name: "benchmark", targets: ["benchmark"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
    ],
    targets: [
        .target(
            name: "Haitaka",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "Sources/Haitaka"
        ),
        .executableTarget(
            name: "train",
            dependencies: ["Haitaka", .product(name: "MLX", package: "mlx-swift")],
            path: "script/train",
            exclude: ["run.sh"]
        ),
        .executableTarget(
            name: "pretrain",
            dependencies: ["Haitaka", .product(name: "MLX", package: "mlx-swift")],
            path: "script/pretrain"
        ),
        .executableTarget(name: "transcribe", dependencies: ["Haitaka"], path: "script/transcribe"),
        .executableTarget(name: "mictrans", dependencies: ["Haitaka"], path: "Sources/mictrans"),
        .executableTarget(name: "stage2", dependencies: ["Haitaka"], path: "script/stage2"),
        .executableTarget(name: "kanalm", dependencies: ["Haitaka"], path: "script/kanalm"),
        .executableTarget(
            name: "kanatext",
            dependencies: ["Haitaka", .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                           .product(name: "MLXOptimizers", package: "mlx-swift")],
            path: "script/kanatext"
        ),
        .executableTarget(name: "screen", dependencies: ["Haitaka"], path: "script/screen"),
        .executableTarget(name: "segment", dependencies: ["Haitaka"], path: "script/segment"),
        .executableTarget(
            name: "benchmark",
            dependencies: ["Haitaka"],
            path: "script/benchmark",
            exclude: ["run.sh"]
        ),
        .testTarget(
            name: "HaitakaTests",
            dependencies: ["Haitaka", .product(name: "MLX", package: "mlx-swift")],
            path: "Tests/HaitakaTests"
        )
    ]
)
