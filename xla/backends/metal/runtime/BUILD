load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Pure-C++ artifact type: holds a precompiled MTLComputePipelineState (via
# PipelineStateRef), entry-point name, arity, and launch dimensions. Owned by
# MetalKernelThunk; constructed by MetalCompiler's per-fusion retry loop.
cc_library(
    name = "metal_kernel_artifact",
    hdrs = ["metal_kernel_artifact.h"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor/metal:metal_pso_probe",
    ],
)

# Metal-specific Thunk that dispatches via a precompiled PSO. Bridges back
# from the opaque PipelineStateRef to id<MTLComputePipelineState> inside the
# .mm and reuses the existing MetalKernel / MetalStream launch path.
objc_library(
    name = "metal_kernel_thunk",
    srcs = ["metal_kernel_thunk.mm"],
    hdrs = ["metal_kernel_thunk.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_kernel_artifact",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/metal:metal_executor_header",
        "//xla/stream_executor/metal:metal_kernel",
        "//xla/stream_executor/metal:metal_pso_probe",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)
