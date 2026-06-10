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

#include "xla/backends/metal/transforms/metal_graph_support.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace metal {
namespace {

bool IsSupportedElementType(PrimitiveType type,
                            const MetalGraphCapabilities& caps) {
  switch (type) {
    case F32:
    case F16:
    case PRED:
      return true;
    case BF16:
      return caps.supports_bf16;
    default:
      return false;
  }
}

// Dense, static, array-shaped, supported element type.
bool IsSupportedShape(const Shape& shape, const MetalGraphCapabilities& caps) {
  return shape.IsArray() && shape.is_static() &&
         IsSupportedElementType(shape.element_type(), caps);
}

bool AllShapesSupported(const HloInstruction& instr,
                        const MetalGraphCapabilities& caps) {
  if (!IsSupportedShape(instr.shape(), caps)) {
    return false;
  }
  for (const HloInstruction* operand : instr.operands()) {
    if (!IsSupportedShape(operand->shape(), caps)) {
      return false;
    }
  }
  return true;
}

bool IsTranslatableDot(const HloInstruction& instr,
                       const MetalGraphCapabilities& caps) {
  const auto* dot = Cast<HloDotInstruction>(&instr);
  const PrimitiveType lhs = dot->operand(0)->shape().element_type();
  const PrimitiveType rhs = dot->operand(1)->shape().element_type();
  const PrimitiveType out = dot->shape().element_type();
  if (lhs != rhs) {
    return false;
  }
  const bool float_in = lhs == F32 || lhs == F16 || (lhs == BF16 && caps.supports_bf16);
  const bool float_out =
      out == F32 || out == F16 || (out == BF16 && caps.supports_bf16);
  // Same-type dots, or f16/bf16 inputs accumulating into f32.
  if (!float_in || !float_out || (out != lhs && out != F32)) {
    return false;
  }
  switch (dot->precision_config().algorithm()) {
    case PrecisionConfig::ALG_UNSET:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      break;
    default:
      return false;
  }
  return true;
}

bool IsTranslatableConvolution(const HloInstruction& instr) {
  const ConvolutionDimensionNumbers& dnums =
      instr.convolution_dimension_numbers();
  if (dnums.input_spatial_dimensions_size() != 2) {
    return false;
  }
  if (instr.batch_group_count() != 1) {
    return false;
  }
  const PrimitiveType in = instr.operand(0)->shape().element_type();
  const PrimitiveType kern = instr.operand(1)->shape().element_type();
  const PrimitiveType out = instr.shape().element_type();
  if (in != kern || (out != in && out != F32)) {
    return false;
  }
  if (in != F32 && in != F16) {
    return false;
  }
  for (const WindowDimension& dim : instr.window().dimensions()) {
    // Base dilation is a transposed conv; window reversal has no MPSGraph
    // descriptor equivalent.
    if (dim.window_reversal() || dim.base_dilation() != 1) {
      return false;
    }
  }
  return true;
}

// Single-operand reduce whose combiner is a float monoid (add/mul/max/min of
// its two parameters) and whose init value is that monoid's identity, so the
// fixed-menu MPSGraph reduction computes the same fold.
bool IsTranslatableReduce(const HloInstruction& instr,
                          const MetalGraphCapabilities& caps) {
  if (instr.operand_count() != 2) {
    return false;
  }
  const PrimitiveType type = instr.shape().element_type();
  if (type == PRED || !IsSupportedElementType(type, caps)) {
    return false;
  }
  const HloComputation* combiner = instr.to_apply();
  const HloInstruction* root = combiner->root_instruction();
  if (root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return false;
  }
  const HloInstruction* init = instr.operand(1);
  if (init->opcode() != HloOpcode::kConstant) {
    return false;
  }
  const Literal& literal = Cast<HloConstantInstruction>(init)->literal();
  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return literal == LiteralUtil::Zero(type);
    case HloOpcode::kMultiply:
      return literal == LiteralUtil::One(type);
    case HloOpcode::kMaximum:
      return literal == LiteralUtil::MinValue(type);
    case HloOpcode::kMinimum:
      return literal == LiteralUtil::MaxValue(type);
    default:
      return false;
  }
}

}  // namespace

bool IsMetalGraphFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto config = instr.backend_config<gpu::GpuBackendConfig>();
  return config.ok() &&
         config->fusion_backend_config().kind() == gpu::kMetalGraphFusionKind;
}

bool IsMpsGraphTranslatable(const HloInstruction& instr,
                            const MetalGraphCapabilities& caps) {
  if (instr.HasSideEffect()) {
    return false;
  }
  switch (instr.opcode()) {
    // Anchors.
    case HloOpcode::kDot:
      return AllShapesSupported(instr, caps) && IsTranslatableDot(instr, caps);
    case HloOpcode::kConvolution:
      return AllShapesSupported(instr, caps) && IsTranslatableConvolution(instr);

    // Reduction (autotuner path and grow-through; never a region root chosen
    // by the partitioner).
    case HloOpcode::kReduce:
      return AllShapesSupported(instr, caps) &&
             IsTranslatableReduce(instr, caps);

    // Elementwise unary.
    case HloOpcode::kAbs:
    case HloOpcode::kCeil:
    case HloOpcode::kCos:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kNegate:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
    // Elementwise binary.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kSubtract:
    // Predicate logic.
    case HloOpcode::kAnd:
    case HloOpcode::kCompare:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    // Ternary.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    // Data movement (interior values are logical; layout differences only
    // matter at region boundaries).
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
      return AllShapesSupported(instr, caps);

    case HloOpcode::kConstant:
      return IsSupportedShape(instr.shape(), caps) &&
             ShapeUtil::IsScalar(instr.shape());

    case HloOpcode::kParameter:
      return IsSupportedShape(instr.shape(), caps);

    default:
      return false;
  }
}

absl::Status RegionIsTranslatable(const HloComputation& computation,
                                  const MetalGraphCapabilities& caps) {
  for (const HloInstruction* instr : computation.instructions()) {
    if (!IsMpsGraphTranslatable(*instr, caps)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Instruction is not MPSGraph-translatable: ",
                       instr->ToString()));
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace xla
