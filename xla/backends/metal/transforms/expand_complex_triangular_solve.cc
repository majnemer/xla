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

#include "xla/backends/metal/transforms/expand_complex_triangular_solve.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/primitive_util.h"
#include "xla/service/triangular_solve_expander.h"

namespace xla::metal {

bool MetalExpandComplexTriangularSolve::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (!TriangularSolveExpander::InstructionMatchesPattern(instruction)) {
    return false;
  }
  return primitive_util::IsComplexType(
      instruction->operand(0)->shape().element_type());
}

}  // namespace xla::metal
