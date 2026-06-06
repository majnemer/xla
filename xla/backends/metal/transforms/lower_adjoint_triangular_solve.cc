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

#include "xla/backends/metal/transforms/lower_adjoint_triangular_solve.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla::metal {

absl::StatusOr<bool> MetalLowerAdjointTriangularSolve::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // Collect first so we don't iterate over the conjugate ops we inserted.
    std::vector<HloInstruction*> targets;
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kTriangularSolve) continue;
      if (instr->triangular_solve_options().transpose_a() !=
          TriangularSolveOptions::ADJOINT) {
        continue;
      }
      targets.push_back(instr);
    }
    for (HloInstruction* trsm : targets) {
      HloInstruction* a = trsm->mutable_operand(0);
      const PrimitiveType element_type = a->shape().element_type();
      if (primitive_util::IsComplexType(element_type)) {
        // Build conj(A) = Complex(Real(A), Negate(Imag(A))). The component
        // type is the real-precision sibling (f32 for c64, f64 for c128).
        const PrimitiveType component_type =
            primitive_util::ComplexComponentType(element_type);
        const Shape real_shape =
            ShapeUtil::ChangeElementType(a->shape(), component_type);
        HloInstruction* real_a = comp->AddInstruction(
            HloInstruction::CreateUnary(real_shape, HloOpcode::kReal, a));
        HloInstruction* imag_a = comp->AddInstruction(
            HloInstruction::CreateUnary(real_shape, HloOpcode::kImag, a));
        HloInstruction* neg_imag = comp->AddInstruction(
            HloInstruction::CreateUnary(real_shape, HloOpcode::kNegate,
                                        imag_a));
        HloInstruction* conj_a = comp->AddInstruction(
            HloInstruction::CreateBinary(a->shape(), HloOpcode::kComplex,
                                         real_a, neg_imag));
        TF_RETURN_IF_ERROR(trsm->ReplaceOperandWith(0, conj_a));
      }
      // Both real and complex paths land here: produce a fresh trsm with
      // transpose_a == TRANSPOSE. For real this is a no-op semantically; for
      // complex, conj(A)^T == A^H, so TRANSPOSE on the conjugated operand
      // computes the same operator ADJOINT requested.
      TriangularSolveOptions opts = trsm->triangular_solve_options();
      opts.set_transpose_a(TriangularSolveOptions::TRANSPOSE);
      TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
          trsm, HloInstruction::CreateTriangularSolve(
                    trsm->shape(), trsm->mutable_operand(0),
                    trsm->mutable_operand(1), opts)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla::metal
