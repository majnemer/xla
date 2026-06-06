"""Metal-platform specific StreamExecutor support code."""

load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

# Pure-C++ identifier; safe to build on any host. Runtime targets that
# link Metal.framework will be added later and select()-gated to macOS.
cc_library(
    name = "metal_platform_id",
    srcs = ["metal_platform_id.cc"],
    hdrs = ["metal_platform_id.h"],
    deps = ["//xla/stream_executor:platform_id"],
)

tf_proto_library(
    name = "metal_compute_capability_proto",
    srcs = ["metal_compute_capability.proto"],
    make_default_target_header_only = True,
)

# Pure-C++ capability struct. Mirrors MTLGPUFamily values into a plain int
# at the runtime boundary so headers stay free of Obj-C types.
cc_library(
    name = "metal_compute_capability",
    hdrs = ["metal_compute_capability.h"],
    deps = [
        ":metal_compute_capability_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

# Per-device StreamExecutor — header-only target, split out so that other
# Obj-C++ targets in this directory (e.g. metal_stream) can depend on the
# header without pulling in the impl. The pattern mirrors cuda_executor /
# cuda_executor_lib in the cuda/ BUILD.
objc_library(
    name = "metal_executor_header",
    hdrs = ["metal_executor.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_allocator",
        ":metal_compute_capability",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

# MTLBuffer-backed allocator. Shared-mode buffers on Apple Silicon are
# unified memory, so the buffer's contents pointer doubles as the host-side
# address. Obj-C++; macOS-only.
objc_library(
    name = "metal_allocator",
    srcs = ["metal_allocator.mm"],
    hdrs = ["metal_allocator.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

# Per-device StreamExecutor implementation. Obj-C++; macOS-only. The header
# imports <Metal/Metal.h>, so consumers must be Obj-C++ as well — confined to
# the metal/ subdirectory by convention.
objc_library(
    name = "metal_executor",
    srcs = ["metal_executor.mm"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_compute_capability",
        ":metal_event",
        ":metal_executor_header",
        ":metal_kernel",
        ":metal_platform_id",
        ":metal_stream",
        ":metal_timer",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:event",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args_packing_spec",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

# Kernel wrapper around an MTLComputePipelineState. Obj-C++; macOS-only.
objc_library(
    name = "metal_kernel",
    srcs = ["metal_kernel.mm"],
    hdrs = ["metal_kernel.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_executor_header",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# MTLSharedEvent-backed Event implementation. Obj-C++; macOS-only.
objc_library(
    name = "metal_event",
    srcs = ["metal_event.mm"],
    hdrs = ["metal_event.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_executor_header",
        "//xla/stream_executor:event",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

# EventBasedTimer for Metal. Builds two timing-capable MetalEvents and reads
# GPU time off the underlying MTLCommandBuffers via GPUStartTime/GPUEndTime.
objc_library(
    name = "metal_timer",
    srcs = ["metal_timer.mm"],
    hdrs = ["metal_timer.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_event",
        ":metal_executor_header",
        "//xla/stream_executor:event_based_timer",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

# Per-stream wrapper around an MTLCommandQueue. Obj-C++; macOS-only.
objc_library(
    name = "metal_stream",
    srcs = ["metal_stream.mm"],
    hdrs = ["metal_stream.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_allocator",
        ":metal_event",
        ":metal_executor_header",
        ":metal_kernel",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

# End-to-end smoke test: compiles a tiny MSL kernel, allocates a buffer,
# launches the kernel, reads back, and verifies the result. Pulls in the
# metal_platform initializer via deps so PlatformManager::PlatformWithName
# resolves "METAL" at runtime.
cc_test(
    name = "metal_smoke_test",
    srcs = ["metal_smoke_test.cc"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_platform",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Platform plugin. Obj-C++ implementation; macOS-only. Public header is
# pure C++ — Metal/Foundation types do not appear in metal_platform.h.
objc_library(
    name = "metal_platform",
    srcs = ["metal_platform.mm"],
    hdrs = ["metal_platform.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
    deps = [
        ":metal_executor",
        ":metal_fft",
        ":metal_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

# MPSGraph-backed FFT plugin. Registers itself with the PluginRegistry
# under kMetalPlatformId at module load; pulled in via :metal_platform
# above so the registration runs whenever the Metal platform is loaded.
objc_library(
    name = "metal_fft",
    srcs = ["metal_fft.mm"],
    hdrs = ["metal_fft.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_allocator",
        ":metal_executor",
        ":metal_platform_id",
        ":metal_stream",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
