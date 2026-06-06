load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Metal-specific HLO module passes. These run inside MetalCompiler's
# pipeline; other GPU backends do not see them.
#
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
