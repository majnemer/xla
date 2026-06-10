load("//xla/tsl/platform:rules_cc.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Metal-specific HLO module passes. These run inside MetalCompiler's
# pipeline; other GPU backends do not see them.

# Translatability gate for MPSGraph-bound regions. Pure C++ — the single
# source of truth consulted by the partitioner, the autotuner backend, and
# tests. The Obj-C++ emitter must accept everything this gate accepts.
cc_library(
    name = "metal_graph_support",
    srcs = ["metal_graph_support.cc"],
    hdrs = ["metal_graph_support.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# Forms __metal_graph kCustom fusions around translatable dot/conv anchors.
# Runs pre-fusion, post-layout-normalization (see
# GpuCompiler::AddGraphCompilerFusionPasses).
cc_library(
    name = "metal_graph_partitioner",
    srcs = ["metal_graph_partitioner.cc"],
    hdrs = ["metal_graph_partitioner.h"],
    deps = [
        ":metal_graph_support",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "metal_graph_partitioner_test",
    srcs = ["metal_graph_partitioner_test.cc"],
    deps = [
        ":metal_graph_partitioner",
        ":metal_graph_support",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Expand complex-typed kTriangularSolve into the standard MAGMA
# block-inversion sequence. MPSMatrixSolveTriangular asserts F32-only at
# encode time; the expander reuses XLA's stock TriangularSolveExpander
# but gates on complex element type so F32 keeps the fast MPS path.
cc_library(
    name = "expand_complex_triangular_solve",
    srcs = ["expand_complex_triangular_solve.cc"],
    hdrs = ["expand_complex_triangular_solve.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:triangular_solve_expander",
        "@com_google_absl//absl/strings",
    ],
)
