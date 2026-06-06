load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Metal-specific HLO module passes. These run inside MetalCompiler's
# pipeline; other GPU backends do not see them.
cc_library(
    name = "lower_adjoint_triangular_solve",
    srcs = ["lower_adjoint_triangular_solve.cc"],
    hdrs = ["lower_adjoint_triangular_solve.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
