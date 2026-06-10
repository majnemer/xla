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

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "xla/backends/metal/codegen/metal_graph_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/metal/runtime/metal_graph_artifact.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace metal {
namespace {

absl::StatusOr<MPSDataType> MpsDataTypeFor(PrimitiveType type) {
  switch (type) {
    case F32:
      return MPSDataTypeFloat32;
    case F16:
      return MPSDataTypeFloat16;
    case PRED:
      return MPSDataTypeBool;
    case BF16:
      if (@available(macOS 14.0, *)) {
        return MPSDataTypeBFloat16;
      }
      return absl::InternalError("BF16 requires macOS 14+");
    default:
      return absl::InternalError(
          absl::StrCat("MetalGraphEmitter: unsupported element type ",
                       PrimitiveType_Name(type)));
  }
}

// Logical dimension sizes; rank-0 maps to [1] so every tensor stays
// MPSNDArray-representable (broadcast/reshape recipes account for this).
std::vector<int64_t> LogicalDims(const Shape& shape) {
  if (shape.dimensions().empty()) {
    return {1};
  }
  return std::vector<int64_t>(shape.dimensions().begin(),
                              shape.dimensions().end());
}

// Dimension sizes in physical (slowest-to-fastest) order: the dense
// row-major interpretation of the XLA buffer.
std::vector<int64_t> PhysicalDims(const Shape& shape) {
  if (shape.dimensions().empty()) {
    return {1};
  }
  std::vector<int64_t> dims;
  dims.reserve(shape.dimensions().size());
  const auto& minor_to_major = shape.layout().minor_to_major();
  for (auto it = minor_to_major.rbegin(); it != minor_to_major.rend(); ++it) {
    dims.push_back(shape.dimensions(*it));
  }
  return dims;
}

// Permutation p with logical[j] = physical[p[j]]; std::nullopt if identity
// (descending layout — the common case post-LayoutNormalization).
std::optional<std::vector<int64_t>> PhysicalToLogicalPerm(const Shape& shape) {
  const int64_t rank = shape.dimensions().size();
  if (rank <= 1) {
    return std::nullopt;
  }
  const auto& minor_to_major = shape.layout().minor_to_major();
  // physical position k holds logical dimension minor_to_major[rank - 1 - k].
  std::vector<int64_t> position_of_logical(rank);
  for (int64_t k = 0; k < rank; ++k) {
    position_of_logical[minor_to_major[rank - 1 - k]] = k;
  }
  bool identity = true;
  for (int64_t j = 0; j < rank; ++j) {
    identity &= position_of_logical[j] == j;
  }
  if (identity) {
    return std::nullopt;
  }
  return position_of_logical;
}

std::vector<int64_t> InversePermutation(absl::Span<const int64_t> perm) {
  std::vector<int64_t> inverse(perm.size());
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    inverse[perm[i]] = i;
  }
  return inverse;
}

NSArray<NSNumber*>* ToNSShape(absl::Span<const int64_t> dims) {
  NSMutableArray<NSNumber*>* shape =
      [NSMutableArray arrayWithCapacity:dims.size()];
  for (int64_t d : dims) {
    [shape addObject:@(d)];
  }
  return shape;
}

NSArray<NSNumber*>* ToNSNumbers(absl::Span<const int64_t> values) {
  return ToNSShape(values);
}

class GraphBuilder {
 public:
  GraphBuilder(MPSGraph* graph, const HloFusionInstruction& fusion)
      : graph_(graph), fusion_(fusion) {}

  absl::StatusOr<std::unique_ptr<MetalGraphArtifact>> Build(
      id<MTLDevice> device) {
    const HloComputation* body = fusion_.fused_instructions_computation();

    auto artifact = std::make_unique<MetalGraphArtifact>();
    artifact->feeds.resize(body->num_parameters());
    NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
        [NSMutableDictionary dictionary];

    // Parameters: placeholders bind in physical order (the dense view of the
    // operand buffer); an entry transpose recovers logical order.
    for (int64_t i = 0; i < body->num_parameters(); ++i) {
      const HloInstruction* param = body->parameter_instruction(i);
      const Shape& operand_shape = fusion_.operand(i)->shape();
      TF_ASSIGN_OR_RETURN(MPSDataType dtype,
                          MpsDataTypeFor(operand_shape.element_type()));
      std::vector<int64_t> physical = PhysicalDims(operand_shape);
      NSArray<NSNumber*>* ns_physical = ToNSShape(physical);
      MPSGraphTensor* placeholder = [graph_
          placeholderWithShape:ns_physical
                      dataType:dtype
                          name:[NSString stringWithFormat:@"p%lld",
                                                          (long long)i]];
      feeds[placeholder] =
          [[MPSGraphShapedType alloc] initWithShape:ns_physical
                                           dataType:dtype];
      artifact->feeds[i].physical_dims = std::move(physical);
      artifact->feeds[i].mps_data_type = static_cast<uint32_t>(dtype);

      MPSGraphTensor* logical = placeholder;
      if (auto perm = PhysicalToLogicalPerm(operand_shape)) {
        logical = [graph_ transposeTensor:placeholder
                              permutation:ToNSNumbers(*perm)
                                     name:nil];
      }
      tensors_[param] = logical;
    }

    for (const HloInstruction* instr : body->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kParameter) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(MPSGraphTensor * tensor, Emit(instr));
      tensors_[instr] = tensor;
    }

    // Result: permute back to the physical order of the fusion's buffer.
    const Shape& result_shape = fusion_.shape();
    TF_ASSIGN_OR_RETURN(MPSDataType result_dtype,
                        MpsDataTypeFor(result_shape.element_type()));
    MPSGraphTensor* result = tensors_.at(body->root_instruction());
    if (auto perm = PhysicalToLogicalPerm(result_shape)) {
      result = [graph_ transposeTensor:result
                           permutation:ToNSNumbers(InversePermutation(*perm))
                                  name:nil];
    }
    // Pure data-movement graphs (a solo transpose) otherwise compile to a
    // view of the input placeholder, and MPSGraphExecutable ELIDES the
    // result write when the destination MPSNDArray shares its MTLBuffer
    // with that placeholder's binding — the alias check is buffer-granular
    // and ignores the differing offsets. Under BFC pooling, feed and result
    // routinely share one pool chunk, leaving the result slice unwritten.
    // An identity forces a materializing kernel, which always writes.
    result = [graph_ identityWithTensor:result name:nil];
    artifact->result.physical_dims = PhysicalDims(result_shape);
    artifact->result.mps_data_type = static_cast<uint32_t>(result_dtype);

    MPSGraphCompilationDescriptor* compile_desc =
        [[MPSGraphCompilationDescriptor alloc] init];
    MPSGraphDevice* graph_device =
        device != nil ? [MPSGraphDevice deviceWithMTLDevice:device] : nil;
    MPSGraphExecutable* executable =
        [graph_ compileWithDevice:graph_device
                            feeds:feeds
                    targetTensors:@[ result ]
                 targetOperations:nil
            compilationDescriptor:compile_desc];
    if (executable == nil) {
      return absl::InternalError(
          absl::StrCat("MetalGraphEmitter: MPSGraph compile returned nil for ",
                       fusion_.name()));
    }
    artifact->executable = stream_executor::metal::GraphExecutableRef::Wrap(
        (__bridge void*)executable);
    return artifact;
  }

 private:
  MPSGraphTensor* Tensor(const HloInstruction* instr) {
    return tensors_.at(instr);
  }

  absl::StatusOr<MPSGraphTensor*> Emit(const HloInstruction* instr) {
    switch (instr->opcode()) {
      case HloOpcode::kConstant:
        return EmitConstant(instr);
      case HloOpcode::kDot:
        return EmitDot(instr);
      case HloOpcode::kConvolution:
        return EmitConvolution(instr);
      case HloOpcode::kReduce:
        return EmitReduce(instr);
      case HloOpcode::kBroadcast:
        return EmitBroadcast(instr);
      case HloOpcode::kReshape:
        return [graph_ reshapeTensor:Tensor(instr->operand(0))
                           withShape:ToNSShape(LogicalDims(instr->shape()))
                                name:nil];
      case HloOpcode::kTranspose:
        return [graph_ transposeTensor:Tensor(instr->operand(0))
                           permutation:ToNSNumbers(instr->dimensions())
                                  name:nil];
      case HloOpcode::kBitcast:
        return EmitBitcast(instr);
      case HloOpcode::kCopy:
        return Tensor(instr->operand(0));
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(MPSDataType dtype,
                            MpsDataTypeFor(instr->shape().element_type()));
        return [graph_ castTensor:Tensor(instr->operand(0))
                           toType:dtype
                             name:nil];
      }
      case HloOpcode::kSelect:
        return [graph_ selectWithPredicateTensor:Tensor(instr->operand(0))
                             truePredicateTensor:Tensor(instr->operand(1))
                            falsePredicateTensor:Tensor(instr->operand(2))
                                            name:nil];
      case HloOpcode::kClamp:
        return [graph_ clampWithTensor:Tensor(instr->operand(1))
                        minValueTensor:Tensor(instr->operand(0))
                        maxValueTensor:Tensor(instr->operand(2))
                                  name:nil];
      case HloOpcode::kCompare:
        return EmitCompare(instr);
      default:
        break;
    }

    if (instr->operand_count() == 1) {
      return EmitUnary(instr);
    }
    if (instr->operand_count() == 2) {
      return EmitBinary(instr);
    }
    return absl::InternalError(absl::StrCat(
        "MetalGraphEmitter: gate/emitter divergence, untranslatable op: ",
        instr->ToString()));
  }

  absl::StatusOr<MPSGraphTensor*> EmitConstant(const HloInstruction* instr) {
    const Literal& literal = Cast<HloConstantInstruction>(instr)->literal();
    std::optional<double> value = literal.GetAsDouble({});
    if (!value.has_value()) {
      return absl::InternalError(absl::StrCat(
          "MetalGraphEmitter: non-scalar or non-numeric constant: ",
          instr->ToString()));
    }
    TF_ASSIGN_OR_RETURN(MPSDataType dtype,
                        MpsDataTypeFor(instr->shape().element_type()));
    return [graph_ constantWithScalar:*value
                                shape:ToNSShape(LogicalDims(instr->shape()))
                             dataType:dtype];
  }

  absl::StatusOr<MPSGraphTensor*> EmitUnary(const HloInstruction* instr) {
    MPSGraphTensor* in = Tensor(instr->operand(0));
    switch (instr->opcode()) {
      case HloOpcode::kAbs:
        return [graph_ absoluteWithTensor:in name:nil];
      case HloOpcode::kCeil:
        return [graph_ ceilWithTensor:in name:nil];
      case HloOpcode::kCos:
        return [graph_ cosWithTensor:in name:nil];
      case HloOpcode::kErf:
        return [graph_ erfWithTensor:in name:nil];
      case HloOpcode::kExp:
        return [graph_ exponentWithTensor:in name:nil];
      case HloOpcode::kExpm1: {
        MPSGraphTensor* exp = [graph_ exponentWithTensor:in name:nil];
        return [graph_ subtractionWithPrimaryTensor:exp
                                    secondaryTensor:OnesLike(instr)
                                               name:nil];
      }
      case HloOpcode::kFloor:
        return [graph_ floorWithTensor:in name:nil];
      case HloOpcode::kLog:
        return [graph_ logarithmWithTensor:in name:nil];
      case HloOpcode::kLog1p: {
        MPSGraphTensor* one_plus =
            [graph_ additionWithPrimaryTensor:in
                              secondaryTensor:OnesLike(instr)
                                         name:nil];
        return [graph_ logarithmWithTensor:one_plus name:nil];
      }
      case HloOpcode::kLogistic:
        return [graph_ sigmoidWithTensor:in name:nil];
      case HloOpcode::kNegate:
        return [graph_ negativeWithTensor:in name:nil];
      case HloOpcode::kNot:
        return [graph_ notWithTensor:in name:nil];
      case HloOpcode::kRsqrt: {
        MPSGraphTensor* sqrt = [graph_ squareRootWithTensor:in name:nil];
        return [graph_ reciprocalWithTensor:sqrt name:nil];
      }
      case HloOpcode::kSign:
        return [graph_ signWithTensor:in name:nil];
      case HloOpcode::kSin:
        return [graph_ sinWithTensor:in name:nil];
      case HloOpcode::kSqrt:
        return [graph_ squareRootWithTensor:in name:nil];
      case HloOpcode::kTanh:
        return [graph_ tanhWithTensor:in name:nil];
      default:
        return absl::InternalError(absl::StrCat(
            "MetalGraphEmitter: gate/emitter divergence, unary op: ",
            instr->ToString()));
    }
  }

  absl::StatusOr<MPSGraphTensor*> EmitBinary(const HloInstruction* instr) {
    MPSGraphTensor* lhs = Tensor(instr->operand(0));
    MPSGraphTensor* rhs = Tensor(instr->operand(1));
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
        return [graph_ additionWithPrimaryTensor:lhs
                                 secondaryTensor:rhs
                                            name:nil];
      case HloOpcode::kSubtract:
        return [graph_ subtractionWithPrimaryTensor:lhs
                                    secondaryTensor:rhs
                                               name:nil];
      case HloOpcode::kMultiply:
        return [graph_ multiplicationWithPrimaryTensor:lhs
                                       secondaryTensor:rhs
                                                  name:nil];
      case HloOpcode::kDivide:
        return [graph_ divisionWithPrimaryTensor:lhs
                                 secondaryTensor:rhs
                                            name:nil];
      // XLA maximum/minimum propagate NaNs; the plain MPSGraph ops follow
      // IEEE maxNum/minNum and return the non-NaN operand.
      case HloOpcode::kMaximum:
        return [graph_ maximumWithNaNPropagationWithPrimaryTensor:lhs
                                                  secondaryTensor:rhs
                                                             name:nil];
      case HloOpcode::kMinimum:
        return [graph_ minimumWithNaNPropagationWithPrimaryTensor:lhs
                                                  secondaryTensor:rhs
                                                             name:nil];
      case HloOpcode::kPower:
        return [graph_ powerWithPrimaryTensor:lhs
                              secondaryTensor:rhs
                                         name:nil];
      case HloOpcode::kAnd:
        return [graph_ logicalANDWithPrimaryTensor:lhs
                                   secondaryTensor:rhs
                                              name:nil];
      case HloOpcode::kOr:
        return [graph_ logicalORWithPrimaryTensor:lhs
                                  secondaryTensor:rhs
                                             name:nil];
      case HloOpcode::kXor:
        return [graph_ logicalXORWithPrimaryTensor:lhs
                                   secondaryTensor:rhs
                                              name:nil];
      default:
        return absl::InternalError(absl::StrCat(
            "MetalGraphEmitter: gate/emitter divergence, binary op: ",
            instr->ToString()));
    }
  }

  absl::StatusOr<MPSGraphTensor*> EmitCompare(const HloInstruction* instr) {
    MPSGraphTensor* lhs = Tensor(instr->operand(0));
    MPSGraphTensor* rhs = Tensor(instr->operand(1));
    switch (Cast<HloCompareInstruction>(instr)->direction()) {
      case ComparisonDirection::kEq:
        return [graph_ equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
      case ComparisonDirection::kNe:
        return [graph_ notEqualWithPrimaryTensor:lhs
                                 secondaryTensor:rhs
                                            name:nil];
      case ComparisonDirection::kLt:
        return [graph_ lessThanWithPrimaryTensor:lhs
                                 secondaryTensor:rhs
                                            name:nil];
      case ComparisonDirection::kLe:
        return [graph_ lessThanOrEqualToWithPrimaryTensor:lhs
                                          secondaryTensor:rhs
                                                     name:nil];
      case ComparisonDirection::kGt:
        return [graph_ greaterThanWithPrimaryTensor:lhs
                                    secondaryTensor:rhs
                                               name:nil];
      case ComparisonDirection::kGe:
        return [graph_ greaterThanOrEqualToWithPrimaryTensor:lhs
                                             secondaryTensor:rhs
                                                        name:nil];
    }
  }

  absl::StatusOr<MPSGraphTensor*> EmitBroadcast(const HloInstruction* instr) {
    const HloInstruction* operand = instr->operand(0);
    std::vector<int64_t> result_dims = LogicalDims(instr->shape());
    // Place operand dims at their broadcast positions, 1 elsewhere, then let
    // MPSGraph broadcast expand.
    std::vector<int64_t> staged(result_dims.size(), 1);
    for (int64_t i = 0; i < static_cast<int64_t>(instr->dimensions().size());
         ++i) {
      staged[instr->dimensions(i)] = operand->shape().dimensions(i);
    }
    MPSGraphTensor* reshaped =
        [graph_ reshapeTensor:Tensor(operand)
                    withShape:ToNSShape(staged)
                         name:nil];
    return [graph_ broadcastTensor:reshaped
                           toShape:ToNSShape(result_dims)
                              name:nil];
  }

  // Bitcast reinterprets the buffer bytes: go to physical order, reshape to
  // the new physical dims, return to the new logical order. With descending
  // layouts both transposes elide to a plain reshape.
  absl::StatusOr<MPSGraphTensor*> EmitBitcast(const HloInstruction* instr) {
    const Shape& in_shape = instr->operand(0)->shape();
    const Shape& out_shape = instr->shape();
    MPSGraphTensor* t = Tensor(instr->operand(0));
    if (auto perm = PhysicalToLogicalPerm(in_shape)) {
      t = [graph_ transposeTensor:t
                      permutation:ToNSNumbers(InversePermutation(*perm))
                             name:nil];
    }
    t = [graph_ reshapeTensor:t
                    withShape:ToNSShape(PhysicalDims(out_shape))
                         name:nil];
    if (auto perm = PhysicalToLogicalPerm(out_shape)) {
      t = [graph_ transposeTensor:t permutation:ToNSNumbers(*perm) name:nil];
    }
    return t;
  }

  absl::StatusOr<MPSGraphTensor*> EmitDot(const HloInstruction* instr) {
    const auto* dot = Cast<HloDotInstruction>(instr);
    const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
    const Shape& lhs_shape = dot->operand(0)->shape();
    const Shape& rhs_shape = dot->operand(1)->shape();

    auto free_dims = [](const Shape& shape,
                        absl::Span<const int64_t> batch,
                        absl::Span<const int64_t> contract) {
      std::vector<int64_t> free;
      for (int64_t d = 0; d < static_cast<int64_t>(shape.dimensions().size());
           ++d) {
        if (absl::c_find(batch, d) == batch.end() &&
            absl::c_find(contract, d) == contract.end()) {
          free.push_back(d);
        }
      }
      return free;
    };
    std::vector<int64_t> lhs_batch(dnums.lhs_batch_dimensions().begin(),
                                   dnums.lhs_batch_dimensions().end());
    std::vector<int64_t> lhs_contract(
        dnums.lhs_contracting_dimensions().begin(),
        dnums.lhs_contracting_dimensions().end());
    std::vector<int64_t> rhs_batch(dnums.rhs_batch_dimensions().begin(),
                                   dnums.rhs_batch_dimensions().end());
    std::vector<int64_t> rhs_contract(
        dnums.rhs_contracting_dimensions().begin(),
        dnums.rhs_contracting_dimensions().end());
    std::vector<int64_t> lhs_free = free_dims(lhs_shape, lhs_batch,
                                              lhs_contract);
    std::vector<int64_t> rhs_free = free_dims(rhs_shape, rhs_batch,
                                              rhs_contract);

    auto dim_product = [](const Shape& shape, absl::Span<const int64_t> dims) {
      int64_t p = 1;
      for (int64_t d : dims) {
        p *= shape.dimensions(d);
      }
      return p;
    };
    const int64_t b = dim_product(lhs_shape, lhs_batch);
    const int64_t m = dim_product(lhs_shape, lhs_free);
    const int64_t k = dim_product(lhs_shape, lhs_contract);
    const int64_t n = dim_product(rhs_shape, rhs_free);

    auto to_3d = [&](MPSGraphTensor* t, std::vector<int64_t> perm,
                     std::vector<int64_t> shape3d) {
      bool identity = true;
      for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        identity &= perm[i] == i;
      }
      if (!identity) {
        t = [graph_ transposeTensor:t permutation:ToNSNumbers(perm) name:nil];
      }
      return [graph_ reshapeTensor:t withShape:ToNSShape(shape3d) name:nil];
    };
    std::vector<int64_t> lhs_perm = lhs_batch;
    lhs_perm.insert(lhs_perm.end(), lhs_free.begin(), lhs_free.end());
    lhs_perm.insert(lhs_perm.end(), lhs_contract.begin(), lhs_contract.end());
    std::vector<int64_t> rhs_perm = rhs_batch;
    rhs_perm.insert(rhs_perm.end(), rhs_contract.begin(), rhs_contract.end());
    rhs_perm.insert(rhs_perm.end(), rhs_free.begin(), rhs_free.end());

    MPSGraphTensor* lhs = Tensor(dot->operand(0));
    MPSGraphTensor* rhs = Tensor(dot->operand(1));

    // f32 accumulation when the result type or the precision config demand
    // it; MPSGraph publishes no accumulator control, so widen the operands.
    const PrimitiveType in_type = lhs_shape.element_type();
    const PrimitiveType out_type = dot->shape().element_type();
    const PrecisionConfig& precision = dot->precision_config();
    bool wants_f32_accumulation =
        in_type != F32 &&
        (out_type == F32 ||
         precision.algorithm() == PrecisionConfig::ALG_DOT_F16_F16_F32 ||
         precision.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
         absl::c_any_of(precision.operand_precision(), [](int p) {
           return p != PrecisionConfig::DEFAULT;
         }));
    if (wants_f32_accumulation) {
      lhs = [graph_ castTensor:lhs toType:MPSDataTypeFloat32 name:nil];
      rhs = [graph_ castTensor:rhs toType:MPSDataTypeFloat32 name:nil];
    }

    MPSGraphTensor* lhs3d = to_3d(lhs, lhs_perm, {b, m, k});
    MPSGraphTensor* rhs3d = to_3d(rhs, rhs_perm, {b, k, n});
    MPSGraphTensor* product =
        [graph_ matrixMultiplicationWithPrimaryTensor:lhs3d
                                      secondaryTensor:rhs3d
                                                 name:nil];

    TF_ASSIGN_OR_RETURN(MPSDataType result_dtype, MpsDataTypeFor(out_type));
    if (wants_f32_accumulation && out_type != F32) {
      product = [graph_ castTensor:product toType:result_dtype name:nil];
    }
    // XLA dot result order is batch, lhs free, rhs free — exactly [B, M, N].
    return [graph_ reshapeTensor:product
                       withShape:ToNSShape(LogicalDims(dot->shape()))
                            name:nil];
  }

  absl::StatusOr<MPSGraphTensor*> EmitConvolution(
      const HloInstruction* instr) {
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    const Window& window = instr->window();
    const WindowDimension& h = window.dimensions(0);
    const WindowDimension& w = window.dimensions(1);

    auto transpose_to = [&](MPSGraphTensor* t, std::vector<int64_t> perm) {
      bool identity = true;
      for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        identity &= perm[i] == i;
      }
      if (identity) {
        return t;
      }
      return [graph_ transposeTensor:t permutation:ToNSNumbers(perm) name:nil];
    };

    MPSGraphTensor* input = Tensor(instr->operand(0));
    MPSGraphTensor* kernel = Tensor(instr->operand(1));
    const PrimitiveType in_type = instr->operand(0)->shape().element_type();
    const PrimitiveType out_type = instr->shape().element_type();
    if (out_type == F32 && in_type != F32) {
      input = [graph_ castTensor:input toType:MPSDataTypeFloat32 name:nil];
      kernel = [graph_ castTensor:kernel toType:MPSDataTypeFloat32 name:nil];
    }

    MPSGraphTensor* nchw = transpose_to(
        input, {dnums.input_batch_dimension(), dnums.input_feature_dimension(),
                dnums.input_spatial_dimensions(0),
                dnums.input_spatial_dimensions(1)});
    MPSGraphTensor* oihw = transpose_to(
        kernel,
        {dnums.kernel_output_feature_dimension(),
         dnums.kernel_input_feature_dimension(),
         dnums.kernel_spatial_dimensions(0),
         dnums.kernel_spatial_dimensions(1)});

    MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:(NSUInteger)w.stride()
                      strideInY:(NSUInteger)h.stride()
                dilationRateInX:(NSUInteger)w.window_dilation()
                dilationRateInY:(NSUInteger)h.window_dilation()
                         groups:(NSUInteger)instr->feature_group_count()
                    paddingLeft:(NSUInteger)w.padding_low()
                   paddingRight:(NSUInteger)w.padding_high()
                     paddingTop:(NSUInteger)h.padding_low()
                  paddingBottom:(NSUInteger)h.padding_high()
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    if (desc == nil) {
      return absl::InternalError(
          "MetalGraphEmitter: conv descriptor init returned nil");
    }
    MPSGraphTensor* conv = [graph_ convolution2DWithSourceTensor:nchw
                                                   weightsTensor:oihw
                                                      descriptor:desc
                                                            name:nil];

    // conv is NCHW; permute so logical axis j takes its dnums-assigned source.
    const int64_t rank = 4;
    std::vector<int64_t> perm(rank);
    perm[dnums.output_batch_dimension()] = 0;
    perm[dnums.output_feature_dimension()] = 1;
    perm[dnums.output_spatial_dimensions(0)] = 2;
    perm[dnums.output_spatial_dimensions(1)] = 3;
    return transpose_to(conv, perm);
  }

  absl::StatusOr<MPSGraphTensor*> EmitReduce(const HloInstruction* instr) {
    MPSGraphTensor* in = Tensor(instr->operand(0));
    NSArray<NSNumber*>* axes = ToNSNumbers(instr->dimensions());
    MPSGraphTensor* reduced = nil;
    switch (instr->to_apply()->root_instruction()->opcode()) {
      case HloOpcode::kAdd:
        reduced = [graph_ reductionSumWithTensor:in axes:axes name:nil];
        break;
      case HloOpcode::kMultiply:
        reduced = [graph_ reductionProductWithTensor:in axes:axes name:nil];
        break;
      case HloOpcode::kMaximum:
        reduced = [graph_ reductionMaximumWithTensor:in axes:axes name:nil];
        break;
      case HloOpcode::kMinimum:
        reduced = [graph_ reductionMinimumWithTensor:in axes:axes name:nil];
        break;
      default:
        return absl::InternalError(absl::StrCat(
            "MetalGraphEmitter: gate/emitter divergence, reduce combiner: ",
            instr->ToString()));
    }
    // MPSGraph keeps reduced axes as size-1; drop them.
    return [graph_ reshapeTensor:reduced
                       withShape:ToNSShape(LogicalDims(instr->shape()))
                            name:nil];
  }

  MPSGraphTensor* OnesLike(const HloInstruction* instr) {
    MPSDataType dtype = MpsDataTypeFor(instr->shape().element_type()).value();
    return [graph_ constantWithScalar:1.0
                                shape:ToNSShape(LogicalDims(instr->shape()))
                             dataType:dtype];
  }

  MPSGraph* graph_;
  const HloFusionInstruction& fusion_;
  absl::flat_hash_map<const HloInstruction*, MPSGraphTensor*> tensors_;
};

}  // namespace

MetalGraphCapabilities ProbeMetalGraphCapabilities(void* mtl_device) {
  MetalGraphCapabilities caps;
  if (mtl_device == nullptr) {
    return caps;
  }
  // MPSGraph handles bf16 on Metal3 + macOS 14, but a bf16 region's boundary
  // values are produced/consumed by MSL kernels, and the MSL lowering cannot
  // store or convert bf16 yet. Enable once it can:
  //   id<MTLDevice> device = (__bridge id<MTLDevice>)mtl_device;
  //   if (@available(macOS 14.0, *)) {
  //     caps.supports_bf16 = [device supportsFamily:MTLGPUFamilyMetal3];
  //   }
  return caps;
}

absl::StatusOr<std::unique_ptr<MetalGraphArtifact>> BuildMetalGraphArtifact(
    void* mtl_device, const HloFusionInstruction& fusion) {
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    GraphBuilder builder(graph, fusion);
    return builder.Build((__bridge id<MTLDevice>)mtl_device);
  }
}

}  // namespace metal
}  // namespace xla
