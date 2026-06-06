/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_METAL_TRANSFORMS_EXPAND_COMPLEX_TRIANGULAR_SOLVE_H_
#define XLA_BACKENDS_METAL_TRANSFORMS_EXPAND_COMPLEX_TRIANGULAR_SOLVE_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/triangular_solve_expander.h"

namespace xla::metal {

// Expands complex-typed kTriangularSolve into the standard
// MAGMA-style sequence (block diagonal inversion + matmul + select), the
// same expansion CPU and interpreter use. Real-typed trsms are left alone
// so they keep the MPSMatrixSolveTriangular fast path.
//
// MPSMatrixSolveTriangular asserts at encode time that only F32 is
// supported, so without this pass complex Cholesky and complex trsm
// HLOs would hit an Apple framework abort. With this pass, complex trsm
// lowers into matmul + select + complex elementwise ops that the existing
// Metal MLIR fusion pipeline already handles.
class MetalExpandComplexTriangularSolve : public TriangularSolveExpander {
 public:
  explicit MetalExpandComplexTriangularSolve(int64_t block_size = 128)
      : TriangularSolveExpander(block_size) {}

  absl::string_view name() const override {
    return "metal-expand-complex-triangular-solve";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_TRANSFORMS_EXPAND_COMPLEX_TRIANGULAR_SOLVE_H_
