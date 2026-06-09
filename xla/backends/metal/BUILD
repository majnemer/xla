load("//xla/tsl/platform:rules_cc.bzl", "cc_library", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Apple Metal backend compiler. Inherits xla::gpu::GpuCompiler (the post-
# split, LLVM-free base) so Metal shares the GPU-shaped HLO pipeline
# scaffolding without dragging in llvm::Module / PTX / cubin concerns.
cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        "//xla:util",
        "//xla/backends/gpu/autotuner:factory_metal",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/codegen/emitters/transforms:passes",
        "//xla/backends/gpu/runtime:conditional_thunk",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:fft_thunk",
        "//xla/backends/gpu/runtime:infeed_thunk",
        "//xla/backends/gpu/runtime:kernel_thunk",
        "//xla/backends/gpu/runtime:outfeed_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_executor",
        "//xla/backends/gpu/runtime:while_thunk",
        "//xla:xla_data_proto_cc",
        "//xla/backends/metal/codegen:msl_kernel_emitter",
        "//xla/backends/metal/codegen:sort_emitter",
        "//xla/backends/metal/codegen:msl_kernel_source",
        "//xla/backends/metal/codegen/transforms:passes",
        "//xla/backends/metal/runtime:metal_kernel_artifact",
        "//xla/backends/metal/runtime:metal_kernel_thunk",
        "//xla/backends/metal/runtime:metal_triangular_solve_thunk",
        "//xla/backends/metal/transforms:expand_complex_triangular_solve",
        "//xla/codegen:ir_printing",
        "//xla/codegen:mlir_kernel_source",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:float_normalization",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:call_graph",
        "//xla/service:compiler",
        "//xla/service:computation_placer",
        "//xla/service:dump",
        "//xla/service:float_support",
        "//xla/service:gpu_topology",
        "//xla/service:logical_buffer",
        "//xla/service:name_uniquer",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:alias_info",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/metal:metal_device_handle",
        "//xla/stream_executor/metal:metal_executor",
        "//xla/stream_executor/metal:metal_platform_id",
        "//xla/stream_executor/metal:metal_pso_probe",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:maybe_owning",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ComplexToStandard",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = True,  # Contains compiler registration
)

# Registers GpuCollectivesStub under the METAL platform id so
# GpuCollectives::Default succeeds for Metal-targeted GpuExecutables.
# Metal has no real collectives backend; the stub returns Unimplemented
# from every collective op, which is what we want for single-device.
cc_library(
    name = "collectives_registration",
    srcs = ["collectives_registration.cc"],
    deps = [
        "//xla/backends/gpu/collectives:gpu_collectives_stub",
        "//xla/core/collectives:collectives_registry",
    ],
    alwayslink = True,  # Static-initializer registers the stub.
)

cc_test(
    name = "compiler_test",
    srcs = ["compiler_test.cc"],
    deps = [
        ":compiler",
        "//xla/service:compiler",
        "//xla/stream_executor/metal:metal_platform_id",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# End-to-end: MetalCompiler::RunBackend on a parameter-root HLO produces a
# GpuExecutable whose ExecuteAsyncOnStream returns the input parameter
# buffer unchanged.
cc_test(
    name = "compiler_executable_test",
    srcs = ["compiler_executable_test.cc"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":compiler",
        ":collectives_registration",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:service_executable_run_options",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/metal:metal_platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# End-to-end validation that the existing GPU Thunk runtime composes with the
# Metal StreamExecutor surface. Loads MSL via KernelThunk and runs it through
# Initialize / ExecuteOnStream. Macos-only.
cc_test(
    name = "kernel_thunk_test",
    srcs = ["kernel_thunk_test.cc"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:kernel_thunk",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_executor",
        "//xla/backends/gpu/runtime:while_thunk",
        "//xla:executable_run_options",
        "//xla/backends/metal:collectives_registration",
        "//xla/service:shaped_buffer",
        "//xla/service/gpu:gpu_executable",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/service:buffer_assignment",
        "//xla/service:service_executable_run_options",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:tma_metadata",
        "//xla/stream_executor/metal:metal_platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
