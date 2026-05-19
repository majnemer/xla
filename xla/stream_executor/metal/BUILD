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
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

# Per-device StreamExecutor implementation. Obj-C++; macOS-only. The header
# imports <Metal/Metal.h>, so consumers must be Obj-C++ as well — confined to
# the metal/ subdirectory by convention.
objc_library(
    name = "metal_executor",
    srcs = ["metal_executor.mm"],
    hdrs = ["metal_executor.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_compute_capability",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:event",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
