/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/xtile/codegen/dot_algorithms.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/service/algorithm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace xtile {

namespace {

using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;

Type ElementType(Value v) { return mlir::getElementTypeOrSelf(v); }

mlir::stablehlo::Precision XlaPrecisionToStableHloPrecision(
    PrecisionConfig::Precision precision) {
  switch (precision) {
    case PrecisionConfig::DEFAULT:
      return mlir::stablehlo::Precision::DEFAULT;
    case PrecisionConfig::HIGH:
      return mlir::stablehlo::Precision::HIGH;
    case PrecisionConfig::HIGHEST:
      return mlir::stablehlo::Precision::HIGHEST;
    default:
      LOG(FATAL) << "Unsupported precision: " << precision;
  }
}

}  // namespace

namespace {

absl::StatusOr<Value> ScaledDot(mlir::ImplicitLocOpBuilder& b,
                                ScaledDotOperands& operands) {
  mlir::Type lhs_dot_elem_type = getElementTypeOrSelf(operands.lhs.getType());
  mlir::Type rhs_dot_elem_type = getElementTypeOrSelf(operands.rhs.getType());

  Value lhs_scale;
  if (lhs_dot_elem_type != b.getBF16Type()) {
    lhs_scale = Bitcast(b, operands.lhs_scale, b.getI8Type());
  }
  Value rhs_scale;
  if (rhs_dot_elem_type != b.getBF16Type()) {
    rhs_scale = Bitcast(b, operands.rhs_scale, b.getI8Type());
    auto rhs_scale_type = mlir::cast<mlir::ShapedType>(rhs_scale.getType());
    int64_t rank = rhs_scale_type.getRank();
    CHECK_GE(rank, 2) << "RHS scale must be at least rank 2 for scaled dot.";

    std::vector<int64_t> permutation(rank);
    for (int64_t i = 0; i < rank; ++i) {
      permutation[i] = i;
    }
    std::swap(permutation[rank - 2], permutation[rank - 1]);
    rhs_scale = mlir::stablehlo::TransposeOp::create(
        b, rhs_scale, b.getDenseI64ArrayAttr(permutation));
  }

  // When operand type is subbyte size then it is packed along minor dim and for
  // RHS minor dim is not K.
  const auto& lhs_shaped_type =
      mlir::dyn_cast<ShapedType>(operands.lhs.getType());
  const bool rhs_k_pack = lhs_shaped_type.getElementType() !=
                          mlir::Float4E2M1FNType::get(b.getContext());
  auto dot_scaled_op = xtile::DotScaledOp::create(
      b, operands.accumulator.getType(), operands.lhs, operands.rhs, lhs_scale,
      rhs_scale, /*fastMath=*/true, /*lhs_k_pack=*/true, rhs_k_pack,
      operands.dot_dimension_numbers);

  auto add_result =
      mlir::isa<mlir::IntegerType>(
          dot_scaled_op.getResult().getType().getElementType())
          ? mlir::arith::AddIOp::create(b, operands.accumulator, dot_scaled_op)
          : mlir::arith::AddFOp::create(b, operands.accumulator, dot_scaled_op);
  return add_result->getResult(0);
}

namespace {

Value EmitStableHloDotAndAdd(mlir::ImplicitLocOpBuilder& b, Value lhs,
                             Value rhs, Value acc, PrecisionSpec precision_spec,
                             mlir::stablehlo::DotDimensionNumbersAttr dims) {
  auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      b.getContext(), {precision_spec.lhs_operand_precision,
                       precision_spec.rhs_operand_precision});
  auto dot = mlir::stablehlo::DotGeneralOp::create(
      b, acc.getType(), lhs, rhs, dims,
      /*precision_config=*/precision_config,
      /*algorithm=*/
      stablehlo::ConvertDotAlgorithm(precision_spec.algorithm, &b));

  auto add_result =
      mlir::isa<mlir::IntegerType>(dot.getResult().getType().getElementType())
          ? mlir::arith::AddIOp::create(b, acc, dot)
          : mlir::arith::AddFOp::create(b, acc, dot);
  return add_result->getResult(0);
}

}  // namespace

// Returns the `Type` that the dot operands should be casted to if there is a
// clear candidate. Returns `std::nullopt` if no casting is a priori needed.
absl::StatusOr<std::optional<Type>> GetForceOperandsType(
    mlir::ImplicitLocOpBuilder& b, const HloDotInstruction& dot) {
  TF_ASSIGN_OR_RETURN(std::optional<PrimitiveType> operands_type,
                      algorithm_util::GetGemmOperandType(dot));
  if (!operands_type.has_value()) {
    return std::nullopt;
  }
  TF_ASSIGN_OR_RETURN(Type type, PrimitiveTypeToMlirType(b, *operands_type));
  return type;
}

}  // namespace

absl::StatusOr<Type> GetDotAccumulatorType(mlir::ImplicitLocOpBuilder& b,
                                           const HloDotInstruction& dot) {
  TF_ASSIGN_OR_RETURN(PrimitiveType accumulator_type,
                      algorithm_util::GetGemmAccumulatorType(dot));
  return PrimitiveTypeToMlirType(b, accumulator_type);
}

absl::StatusOr<Value> EmitSingleTileDot(mlir::ImplicitLocOpBuilder& b,
                                        const HloDotInstruction& dot,
                                        DotOperands dot_operands) {
  PrecisionConfig::Algorithm algorithm = dot.precision_config().algorithm();
  PrecisionSpec precision_spec{
      algorithm,
      XlaPrecisionToStableHloPrecision(
          dot.precision_config().operand_precision(0)),
      XlaPrecisionToStableHloPrecision(
          dot.precision_config().operand_precision(1))};

  TF_ASSIGN_OR_RETURN(std::optional<Type> force_operands_type,
                      GetForceOperandsType(b, dot));

  TF_ASSIGN_OR_RETURN(Type force_accumulator_type,
                      GetDotAccumulatorType(b, dot));

  if (force_operands_type.has_value()) {
    if (ElementType(dot_operands.lhs) != *force_operands_type) {
      dot_operands.lhs = Cast(b, dot_operands.lhs, *force_operands_type);
    }

    if (ElementType(dot_operands.rhs) != *force_operands_type) {
      dot_operands.rhs = Cast(b, dot_operands.rhs, *force_operands_type);
    }
  }

  if (ElementType(dot_operands.accumulator) != force_accumulator_type) {
    dot_operands.accumulator =
        Cast(b, dot_operands.accumulator, force_accumulator_type);
  }

  mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      xla::stablehlo::ConvertDotDimensionNumbers(dot.dot_dimension_numbers(),
                                                 &b);

  Value result = EmitStableHloDotAndAdd(b, dot_operands.lhs, dot_operands.rhs,
                                        dot_operands.accumulator,
                                        precision_spec, dot_dimension_numbers);

  // TODO(b/393299275): once we've moved on from the legacy emitter, we should
  // make sure that this accumulator type is equal to the one derived here.
  Type outer_accumulator_type = ElementType(dot_operands.accumulator);
  if (ElementType(result) != outer_accumulator_type) {
    result = Cast(b, result, outer_accumulator_type);
  }

  return result;
}

absl::StatusOr<Value> EmitSingleTileScaledDot(
    mlir::ImplicitLocOpBuilder& b, const HloScaledDotInstruction& scaled_dot,
    ScaledDotOperands dot_operands) {
  dot_operands.dot_dimension_numbers =
      ::xla::stablehlo::ConvertDotDimensionNumbers(
          scaled_dot.dot_dimension_numbers(), &b);
  return ScaledDot(b, dot_operands);
}

}  // namespace xtile
}  // namespace xla
