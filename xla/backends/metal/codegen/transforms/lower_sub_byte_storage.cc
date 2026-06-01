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

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/numeric/bits.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace metal {

namespace ma = ::mlir::arith;

#define GEN_PASS_DEF_LOWERSUBBYTESTORAGEPASS
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

namespace {

mlir::IntegerType GetStorageType(mlir::MLIRContext* context) {
  return mlir::IntegerType::get(context, 8);
}

std::optional<int> GetSubByteBitWidth(mlir::Type type) {
  type = mlir::getElementTypeOrSelf(type);
  if (!type.isIntOrFloat()) return std::nullopt;
  int bit_width = type.getIntOrFloatBitWidth();
  if (bit_width == 2 || bit_width == 4) return bit_width;
  return std::nullopt;
}

bool HasSubByteElementType(mlir::Type type) {
  return GetSubByteBitWidth(type).has_value();
}

int64_t PackedElementCount(int64_t logical_element_count, int bit_width) {
  int64_t elements_per_byte = 8 / bit_width;
  return (logical_element_count + elements_per_byte - 1) / elements_per_byte;
}

int SubByteIndexingBits(int bit_width) {
  return absl::bit_width(static_cast<unsigned>(8 / bit_width)) - 1;
}

int IntOrFloatBitWidth(mlir::Type type) {
  return mlir::getElementTypeOrSelf(type).getIntOrFloatBitWidth();
}

mlir::Attribute IntegerValueAttr(mlir::OpBuilder& builder, mlir::Type type,
                                 uint64_t value) {
  mlir::Type element_type = mlir::getElementTypeOrSelf(type);
  int width = mlir::cast<mlir::IntegerType>(element_type).getWidth();
  llvm::APInt int_value(width, value);
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    return mlir::DenseIntElementsAttr::get(shaped, int_value);
  }
  return builder.getIntegerAttr(type, int_value);
}

mlir::Value ConstantIntLike(mlir::ImplicitLocOpBuilder& builder,
                            mlir::Type type, uint64_t value) {
  return ma::ConstantOp::create(
      builder, type, mlir::cast<mlir::TypedAttr>(
                         IntegerValueAttr(builder, type, value)));
}

mlir::Value MaskLowBits(mlir::Value value, int bit_width,
                        mlir::ImplicitLocOpBuilder& builder) {
  return ma::AndIOp::create(builder, value,
                            ConstantIntLike(builder, value.getType(),
                                            (uint64_t{1} << bit_width) - 1));
}

mlir::Value CastIntegerLike(mlir::Value value, mlir::Type dst_type,
                            bool is_signed,
                            mlir::ImplicitLocOpBuilder& builder) {
  if (value.getType() == dst_type) return value;

  int src_width = IntOrFloatBitWidth(value.getType());
  int dst_width = IntOrFloatBitWidth(dst_type);
  if (src_width < dst_width) {
    if (is_signed) return ma::ExtSIOp::create(builder, dst_type, value);
    return ma::ExtUIOp::create(builder, dst_type, value);
  }
  return ma::TruncIOp::create(builder, dst_type, value);
}

mlir::Value SignExtendLowBitsToI8(mlir::Value value, int bit_width,
                                  mlir::ImplicitLocOpBuilder& builder) {
  value = MaskLowBits(value, bit_width, builder);
  uint64_t sign_bit = uint64_t{1} << (bit_width - 1);
  mlir::Value sign = ConstantIntLike(builder, value.getType(), sign_bit);
  return ma::SubIOp::create(builder, ma::XOrIOp::create(builder, value, sign),
                            sign);
}

std::optional<uint64_t> GetRawSubByteBits(mlir::Attribute attr,
                                          int bit_width) {
  uint64_t mask = (uint64_t{1} << bit_width) - 1;
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    return int_attr.getValue().getZExtValue() & mask;
  }
  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    return float_attr.getValue().bitcastToAPInt().getZExtValue() & mask;
  }
  return std::nullopt;
}

mlir::TypedAttr ConvertSubByteAttrToStorage(mlir::TypedAttr attr,
                                            mlir::Type converted_type) {
  std::optional<int> bit_width = GetSubByteBitWidth(attr.getType());
  if (!bit_width) return {};

  auto convert_scalar = [&](mlir::Attribute value) -> mlir::TypedAttr {
    std::optional<uint64_t> bits = GetRawSubByteBits(value, *bit_width);
    if (!bits) return {};
    return mlir::IntegerAttr::get(converted_type, *bits);
  };

  if (!mlir::isa<mlir::ShapedType>(attr.getType())) {
    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      return convert_scalar(int_attr);
    }
    if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
      return convert_scalar(float_attr);
    }
    return {};
  }

  auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  auto converted_shaped_type =
      mlir::dyn_cast<mlir::ShapedType>(converted_type);
  if (!elements || !converted_shaped_type) return {};

  llvm::SmallVector<mlir::Attribute> converted_values;
  mlir::IntegerType i8_type =
      mlir::cast<mlir::IntegerType>(converted_shaped_type.getElementType());

  if (mlir::isa<mlir::RankedTensorType>(attr.getType())) {
    converted_values.assign(converted_shaped_type.getNumElements(),
                            mlir::IntegerAttr::get(i8_type, 0));
    uint64_t mask = (uint64_t{1} << *bit_width) - 1;
    int elements_per_byte = 8 / *bit_width;
    for (auto [index, value] :
         llvm::enumerate(elements.getValues<mlir::Attribute>())) {
      std::optional<uint64_t> bits = GetRawSubByteBits(value, *bit_width);
      if (!bits) return {};
      int64_t byte_index = index / elements_per_byte;
      int shift = (index % elements_per_byte) * *bit_width;
      auto old_value = mlir::cast<mlir::IntegerAttr>(
          converted_values[byte_index]);
      uint64_t packed =
          old_value.getValue().getZExtValue() | ((*bits & mask) << shift);
      converted_values[byte_index] = mlir::IntegerAttr::get(i8_type, packed);
    }
  } else {
    converted_values.reserve(elements.getNumElements());
    for (mlir::Attribute value : elements.getValues<mlir::Attribute>()) {
      std::optional<uint64_t> bits = GetRawSubByteBits(value, *bit_width);
      if (!bits) return {};
      converted_values.push_back(mlir::IntegerAttr::get(i8_type, *bits));
    }
  }

  return mlir::cast<mlir::TypedAttr>(
      mlir::DenseElementsAttr::get(converted_shaped_type, converted_values));
}

class SubByteStorageTypeConverter : public mlir::TypeConverter {
 public:
  explicit SubByteStorageTypeConverter(mlir::MLIRContext* context) {
    addConversion([](mlir::Type type) { return type; });

    addConversion([context](mlir::IntegerType type) -> mlir::Type {
      unsigned width = type.getWidth();
      if (width == 2 || width == 4) return GetStorageType(context);
      return type;
    });

    addConversion([context](mlir::FloatType type) -> mlir::Type {
      int width = type.getWidth();
      if (width == 2 || width == 4) return GetStorageType(context);
      return type;
    });

    addConversion([this, context](mlir::VectorType type) -> mlir::Type {
      if (!HasSubByteElementType(type.getElementType())) return type;
      return type.clone(GetStorageType(context));
    });

    addConversion([context](mlir::RankedTensorType type) -> mlir::Type {
      std::optional<int> bit_width =
          GetSubByteBitWidth(type.getElementType());
      if (!bit_width) return type;
      if (!type.hasStaticShape()) return mlir::Type{};
      return mlir::RankedTensorType::get(
          {PackedElementCount(type.getNumElements(), *bit_width)},
          GetStorageType(context), type.getEncoding());
    });

    addConversion([this](mlir::FunctionType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> inputs;
      llvm::SmallVector<mlir::Type> results;
      if (failed(convertTypes(type.getInputs(), inputs)) ||
          failed(convertTypes(type.getResults(), results))) {
        return type;
      }
      return mlir::FunctionType::get(type.getContext(), inputs, results);
    });
  }
};

std::pair<mlir::Value, mlir::Value> GetPackedByteIndexAndShift(
    mlir::Value logical_index, int bit_width,
    mlir::ImplicitLocOpBuilder& builder) {
  mlir::Value linear = ma::IndexCastUIOp::create(builder, builder.getI64Type(),
                                                 logical_index);
  mlir::Value byte_index_i64 = ma::ShRUIOp::create(
      builder, linear,
      ma::ConstantIntOp::create(builder, builder.getI64Type(),
                                SubByteIndexingBits(bit_width)));
  mlir::Value sub_byte_index = ma::AndIOp::create(
      builder, linear,
      ma::ConstantIntOp::create(builder, builder.getI64Type(),
                                (8 / bit_width) - 1));
  mlir::Value shift_i64 = ma::MulIOp::create(
      builder, sub_byte_index,
      ma::ConstantIntOp::create(builder, builder.getI64Type(), bit_width));
  mlir::Value shift =
      ma::TruncIOp::create(builder, builder.getI8Type(), shift_i64);
  mlir::Value byte_index =
      ma::IndexCastUIOp::create(builder, builder.getIndexType(),
                                byte_index_i64);
  return {byte_index, shift};
}

mlir::Value GetLinearIndex(mlir::ValueRange indices,
                           mlir::ImplicitLocOpBuilder& builder) {
  if (indices.empty()) return ma::ConstantIndexOp::create(builder, 0);
  return indices.front();
}

mlir::Value UnpackSubByteElement(mlir::Value packed_tensor,
                                 mlir::Value logical_index, int bit_width,
                                 mlir::ImplicitLocOpBuilder& builder) {
  auto [byte_index, shift] =
      GetPackedByteIndexAndShift(logical_index, bit_width, builder);
  mlir::Value packed =
      mlir::tensor::ExtractOp::create(builder, packed_tensor, byte_index);
  mlir::Value shifted = ma::ShRUIOp::create(builder, packed, shift);
  return MaskLowBits(shifted, bit_width, builder);
}

mlir::Value InsertSubByteElement(mlir::Value value, mlir::Value packed_tensor,
                                 mlir::Value logical_index, int bit_width,
                                 mlir::ImplicitLocOpBuilder& builder) {
  auto [byte_index, shift] =
      GetPackedByteIndexAndShift(logical_index, bit_width, builder);
  mlir::Value low_bits = MaskLowBits(value, bit_width, builder);
  mlir::Value shifted_value = ma::ShLIOp::create(builder, low_bits, shift);
  mlir::Value mask = ma::ShLIOp::create(
      builder,
      ConstantIntLike(builder, builder.getI8Type(),
                      (uint64_t{1} << bit_width) - 1),
      shift);
  mlir::Value inverse_mask =
      ma::XOrIOp::create(builder, ConstantIntLike(builder, builder.getI8Type(),
                                                 0xff),
                         mask);

  auto atomic_rmw = ::xla::AtomicRMWOp::create(builder, packed_tensor,
                                              byte_index);
  mlir::ImplicitLocOpBuilder body_builder(atomic_rmw.getLoc(),
                                          atomic_rmw.getBodyBuilder());
  mlir::Value preserved =
      ma::AndIOp::create(body_builder, atomic_rmw.getCurrentValue(),
                         inverse_mask);
  mlir::Value merged =
      ma::OrIOp::create(body_builder, preserved, shifted_value);
  ::xla::YieldOp::create(body_builder, merged.getLoc(), merged);
  return atomic_rmw.getResult();
}

template <typename Op>
struct ConvertSubByteBinaryOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width = GetSubByteBitWidth(op.getType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "not a sub-byte result");
    }
    mlir::Type converted_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value result =
        Op::create(builder, converted_type, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, MaskLowBits(result, *bit_width, builder));
    return mlir::success();
  }
};

struct ConvertConstantOp : public mlir::OpConversionPattern<ma::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ConstantOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not a sub-byte constant");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::TypedAttr converted_attr =
        ConvertSubByteAttrToStorage(op.getValue(), converted_type);
    if (!converted_attr) {
      return rewriter.notifyMatchFailure(op, "failed to convert value attr");
    }

    rewriter.replaceOpWithNewOp<ma::ConstantOp>(op, converted_type,
                                                converted_attr);
    return mlir::success();
  }
};

struct ConvertBitcastOp : public mlir::OpConversionPattern<ma::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::BitcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getIn().getType()) &&
        !HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "no sub-byte bitcast operand");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    if (converted_type == adaptor.getIn().getType()) {
      rewriter.replaceOp(op, adaptor.getIn());
      return mlir::success();
    }

    rewriter.replaceOpWithNewOp<ma::BitcastOp>(op, converted_type,
                                               adaptor.getIn());
    return mlir::success();
  }
};

struct ConvertExtUIOp : public mlir::OpConversionPattern<ma::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtUIOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width = GetSubByteBitWidth(op.getIn().getType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "input is not sub-byte");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value value = MaskLowBits(adaptor.getIn(), *bit_width, builder);
    value = CastIntegerLike(value, converted_type, /*is_signed=*/false,
                            builder);
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

struct ConvertExtSIOp : public mlir::OpConversionPattern<ma::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtSIOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width = GetSubByteBitWidth(op.getIn().getType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "input is not sub-byte");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value value =
        SignExtendLowBitsToI8(adaptor.getIn(), *bit_width, builder);
    value =
        CastIntegerLike(value, converted_type, /*is_signed=*/true, builder);
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

struct ConvertTruncIOp : public mlir::OpConversionPattern<ma::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncIOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width = GetSubByteBitWidth(op.getType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "result is not sub-byte");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value value = CastIntegerLike(adaptor.getIn(), converted_type,
                                        /*is_signed=*/false, builder);
    rewriter.replaceOp(op, MaskLowBits(value, *bit_width, builder));
    return mlir::success();
  }
};

struct ConvertCmpIOp : public mlir::OpConversionPattern<ma::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::CmpIOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width = GetSubByteBitWidth(op.getLhs().getType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "operands are not sub-byte");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();
    using Predicate = ma::CmpIPredicate;
    switch (op.getPredicate()) {
      case Predicate::slt:
      case Predicate::sle:
      case Predicate::sgt:
      case Predicate::sge:
        lhs = SignExtendLowBitsToI8(lhs, *bit_width, builder);
        rhs = SignExtendLowBitsToI8(rhs, *bit_width, builder);
        break;
      default:
        lhs = MaskLowBits(lhs, *bit_width, builder);
        rhs = MaskLowBits(rhs, *bit_width, builder);
        break;
    }
    rewriter.replaceOpWithNewOp<ma::CmpIOp>(op, op.getPredicate(), lhs, rhs);
    return mlir::success();
  }
};

struct ConvertSelectOp : public mlir::OpConversionPattern<ma::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::SelectOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<ma::SelectOp>(
        op, converted_type, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

struct ConvertTensorExtract
    : public mlir::OpConversionPattern<mlir::tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width =
        GetSubByteBitWidth(op.getTensor().getType().getElementType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "tensor is not sub-byte");
    }
    if (op.getIndices().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "only rank-0 and rank-1 tensors are supported");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value linear_index = GetLinearIndex(adaptor.getIndices(), builder);
    rewriter.replaceOp(
        op, UnpackSubByteElement(adaptor.getTensor(), linear_index, *bit_width,
                                 builder));
    return mlir::success();
  }
};

struct ConvertTensorInsert
    : public mlir::OpConversionPattern<mlir::tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width =
        GetSubByteBitWidth(op.getDest().getType().getElementType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "dest tensor is not sub-byte");
    }
    if (op.getIndices().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "only rank-0 and rank-1 tensors are supported");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value linear_index = GetLinearIndex(adaptor.getIndices(), builder);
    rewriter.replaceOp(
        op, InsertSubByteElement(adaptor.getScalar(), adaptor.getDest(),
                                 linear_index, *bit_width, builder));
    return mlir::success();
  }
};

template <typename Op>
bool IsSupportedTransfer(Op op) {
  for (bool in_bounds : op.getInBoundsValues()) {
    if (!in_bounds) return false;
  }
  return op.getVectorType().getRank() == 1 && !op.getMask() &&
         op.getPermutationMap().isMinorIdentity();
}

bool IsSupportedTransferRead(mlir::vector::TransferReadOp op) {
  return IsSupportedTransfer(op);
}

bool IsSupportedTransferWrite(mlir::vector::TransferWriteOp op) {
  return IsSupportedTransfer(op);
}

struct ConvertVectorTransferRead
    : public mlir::OpConversionPattern<mlir::vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::TransferReadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width =
        GetSubByteBitWidth(op.getBase().getType().getElementType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "base tensor is not sub-byte");
    }
    if (!IsSupportedTransferRead(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported sub-byte transfer_read");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    auto converted_vector_type =
        mlir::dyn_cast_if_present<mlir::VectorType>(converted_type);
    if (!converted_vector_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value base_index = GetLinearIndex(adaptor.getIndices(), builder);
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(converted_vector_type.getNumElements());
    for (int64_t i = 0; i < converted_vector_type.getNumElements(); ++i) {
      mlir::Value index = base_index;
      if (i != 0) {
        index = ma::AddIOp::create(builder, base_index,
                                   ma::ConstantIndexOp::create(builder, i));
      }
      elements.push_back(UnpackSubByteElement(adaptor.getBase(), index,
                                              *bit_width, builder));
    }

    rewriter.replaceOpWithNewOp<mlir::vector::FromElementsOp>(
        op, converted_vector_type, elements);
    return mlir::success();
  }
};

struct ConvertVectorTransferWrite
    : public mlir::OpConversionPattern<mlir::vector::TransferWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::TransferWriteOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int> bit_width =
        GetSubByteBitWidth(op.getBase().getType().getElementType());
    if (!bit_width) {
      return rewriter.notifyMatchFailure(op, "base tensor is not sub-byte");
    }
    if (!IsSupportedTransferWrite(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported sub-byte transfer_write");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value base_index = GetLinearIndex(adaptor.getIndices(), builder);
    mlir::Value tensor = adaptor.getBase();
    auto vector_type = mlir::cast<mlir::VectorType>(
        adaptor.getValueToStore().getType());
    for (int64_t i = 0; i < vector_type.getNumElements(); ++i) {
      mlir::Value index = base_index;
      if (i != 0) {
        index = ma::AddIOp::create(builder, base_index,
                                   ma::ConstantIndexOp::create(builder, i));
      }
      mlir::Value element = mlir::vector::ExtractOp::create(
          builder, adaptor.getValueToStore(), i);
      tensor = InsertSubByteElement(element, tensor, index, *bit_width,
                                    builder);
    }
    rewriter.replaceOp(op, tensor);
    return mlir::success();
  }
};

struct ConvertVectorExtract
    : public mlir::OpConversionPattern<mlir::vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getSource().getType())) {
      return rewriter.notifyMatchFailure(op, "source is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto new_op = mlir::vector::ExtractOp::create(
        rewriter, op.getLoc(), adaptor.getSource(), op.getMixedPosition());
    if (new_op.getType() != converted_type) {
      return rewriter.notifyMatchFailure(op, "unexpected converted type");
    }
    rewriter.replaceOp(op, new_op.getResult());
    return mlir::success();
  }
};

struct ConvertVectorInsert
    : public mlir::OpConversionPattern<mlir::vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::InsertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getDest().getType())) {
      return rewriter.notifyMatchFailure(op, "dest is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto new_op = mlir::vector::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getValueToStore(), adaptor.getDest(),
        op.getMixedPosition());
    if (new_op.getType() != converted_type) {
      return rewriter.notifyMatchFailure(op, "unexpected converted type");
    }
    rewriter.replaceOp(op, new_op.getResult());
    return mlir::success();
  }
};

struct ConvertVectorFromElements
    : public mlir::OpConversionPattern<mlir::vector::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::FromElementsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::vector::FromElementsOp>(
        op, converted_type, adaptor.getElements());
    return mlir::success();
  }
};

struct ConvertVectorBroadcast
    : public mlir::OpConversionPattern<mlir::vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::BroadcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::vector::BroadcastOp>(
        op, converted_type, adaptor.getSource());
    return mlir::success();
  }
};

struct ConvertPoisonOp : public mlir::OpConversionPattern<mlir::ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ub::PoisonOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasSubByteElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not sub-byte");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, converted_type);
    return mlir::success();
  }
};

class LowerSubByteStoragePass
    : public impl::LowerSubByteStoragePassBase<LowerSubByteStoragePass> {
 public:
  using LowerSubByteStoragePassBase::LowerSubByteStoragePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    SubByteStorageTypeConverter converter(context);

    mlir::RewritePatternSet patterns(context);
    patterns.add<
        ConvertBitcastOp, ConvertCmpIOp, ConvertConstantOp, ConvertExtSIOp,
        ConvertExtUIOp, ConvertPoisonOp, ConvertSelectOp, ConvertTensorExtract,
        ConvertTensorInsert, ConvertTruncIOp, ConvertVectorBroadcast,
        ConvertVectorExtract, ConvertVectorFromElements, ConvertVectorInsert,
        ConvertVectorTransferRead, ConvertVectorTransferWrite,
        ConvertSubByteBinaryOp<ma::AddIOp>,
        ConvertSubByteBinaryOp<ma::AndIOp>,
        ConvertSubByteBinaryOp<ma::MulIOp>, ConvertSubByteBinaryOp<ma::OrIOp>,
        ConvertSubByteBinaryOp<ma::ShLIOp>,
        ConvertSubByteBinaryOp<ma::ShRUIOp>,
        ConvertSubByteBinaryOp<ma::SubIOp>,
        ConvertSubByteBinaryOp<ma::XOrIOp>>(converter, context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);

    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes());
        });
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
      return std::optional<bool>(converter.isLegal(op));
    });
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerSubByteStoragePass() {
  return std::make_unique<LowerSubByteStoragePass>();
}

}  // namespace metal
}  // namespace xla
