load("//xla/tsl/platform:rules_cc.bzl", "cc_library", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Value-type carrier for a JIT-compiled-at-runtime MSL kernel: the source
# text and the entry-point function name within it. Plugged into
# KernelLoaderSpec::CreateMslSourceInMemorySpec by the runtime.
cc_library(
    name = "msl_kernel_source",
    hdrs = ["msl_kernel_source.h"],
    deps = ["//xla/codegen:kernel_source"],
)

# MLIR -> MSL translator. Wedge A: emits the kernel signature only.
# Subsequent wedges fill in body emission for arith / math / scf / tensor /
# gpu / vector / xla_gpu ops.
cc_library(
    name = "translate_to_msl",
    srcs = ["translate_to_msl.cc"],
    hdrs = ["translate_to_msl.h"],
    deps = [
        ":msl_kernel_source",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/service:name_uniquer",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

# Higher-level entry point: runs the Metal-specific MLIR pass pipeline,
# then calls TranslateToMSL. The MetalCompiler's RunBackend will land on
# top of this in a later wedge.
cc_library(
    name = "msl_kernel_emitter",
    srcs = ["msl_kernel_emitter.cc"],
    hdrs = ["msl_kernel_emitter.h"],
    deps = [
        ":msl_kernel_source",
        ":translate_to_msl",
        "//xla/backends/metal/codegen/transforms:passes",
        "//xla/codegen:ir_printing",
        "//xla/hlo/ir:hlo",
        "//xla/service:name_uniquer",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_test(
    name = "msl_kernel_emitter_test",
    srcs = ["msl_kernel_emitter_test.cc"],
    deps = [
        ":msl_kernel_emitter",
        ":msl_kernel_source",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_test(
    name = "translate_to_msl_test",
    srcs = ["translate_to_msl_test.cc"],
    deps = [
        ":msl_kernel_source",
        ":translate_to_msl",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/codegen/emitters/ir:xla",
        "//xla/service:name_uniquer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)
