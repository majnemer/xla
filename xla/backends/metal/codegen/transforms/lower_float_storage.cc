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

#include "llvm/ADT/APFloat.h"
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
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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

namespace xla {
namespace metal {

namespace ma = ::mlir::arith;

#define GEN_PASS_DEF_LOWERFLOATSTORAGEPASS
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

namespace {

mlir::Type CloneWithElementType(mlir::Type type, mlir::Type element_type) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    return vector_type.clone(element_type);
  }
  return element_type;
}

bool IsF8ElementType(mlir::Type type) {
  return mlir::isa<mlir::Float8E3M4Type, mlir::Float8E4M3Type,
                   mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType,
                   mlir::Float8E4M3FNUZType, mlir::Float8E5M2Type,
                   mlir::Float8E5M2FNUZType, mlir::Float8E8M0FNUType>(type);
}

bool IsFloatStorageElementType(mlir::Type type) {
  return mlir::isa<mlir::BFloat16Type>(type) || IsF8ElementType(type);
}

bool HasBf16ElementType(mlir::Type type) {
  return mlir::isa<mlir::BFloat16Type>(mlir::getElementTypeOrSelf(type));
}

bool HasLoweredFloatElementType(mlir::Type type) {
  return IsFloatStorageElementType(mlir::getElementTypeOrSelf(type));
}

mlir::Type ConvertFloatTypeToStorage(mlir::FloatType type) {
  if (IsFloatStorageElementType(type)) {
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  }
  return type;
}

mlir::TypedAttr ConvertFloatAttrToStorage(mlir::TypedAttr attr,
                                          mlir::Type converted_type) {
  mlir::Type element_type = mlir::getElementTypeOrSelf(converted_type);
  if (!mlir::isa<mlir::IntegerType>(element_type)) {
    return {};
  }

  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    return mlir::IntegerAttr::get(element_type,
                                  float_attr.getValue().bitcastToAPInt());
  }

  if (auto elements = mlir::dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
    return mlir::cast<mlir::TypedAttr>(elements.mapValues(
        element_type,
        [](const llvm::APFloat& value) { return value.bitcastToAPInt(); }));
  }

  return {};
}

mlir::Value GetConst(mlir::ImplicitLocOpBuilder& builder, mlir::Type type,
                     mlir::TypedAttr value) {
  if (mlir::isa<mlir::VectorType>(type)) {
    value =
        mlir::SplatElementsAttr::get(mlir::cast<mlir::ShapedType>(type), value);
  }
  return ma::ConstantOp::create(builder, type, value);
}

mlir::Value EmitBF16BitsToF32(mlir::Type dst_ty, mlir::Value in,
                              mlir::ImplicitLocOpBuilder& builder) {
  mlir::Type i32_type =
      CloneWithElementType(in.getType(), builder.getI32Type());
  mlir::Value i32 = ma::ExtUIOp::create(builder, i32_type, in);
  mlir::Value shift =
      GetConst(builder, i32_type, builder.getI32IntegerAttr(16));
  mlir::Value shifted = ma::ShLIOp::create(builder, i32, shift);
  return ma::BitcastOp::create(builder, dst_ty, shifted);
}

mlir::Value EmitF32ToBF16Bits(mlir::Type dst_ty, mlir::Value in,
                              mlir::ImplicitLocOpBuilder& builder) {
  mlir::Type i32_type =
      CloneWithElementType(in.getType(), builder.getI32Type());
  mlir::Value bits = ma::BitcastOp::create(builder, i32_type, in);

  auto cst = [&](uint32_t value) {
    return GetConst(builder, i32_type, builder.getI32IntegerAttr(value));
  };

  mlir::Value shift = cst(16);
  mlir::Value high = ma::ShRUIOp::create(builder, bits, shift);

  mlir::Value abs_bits = ma::AndIOp::create(builder, bits, cst(0x7fffffff));
  mlir::Value is_nan = ma::CmpIOp::create(builder, ma::CmpIPredicate::ugt,
                                          abs_bits, cst(0x7f800000));

  mlir::Value quiet_nan = ma::OrIOp::create(builder, high, cst(0x40));
  mlir::Value lsb = ma::AndIOp::create(builder, high, cst(1));
  mlir::Value rounded = ma::ShRUIOp::create(
      builder,
      ma::AddIOp::create(builder, bits,
                         ma::AddIOp::create(builder, cst(0x7fff), lsb)),
      shift);
  mlir::Value selected =
      ma::SelectOp::create(builder, is_nan, quiet_nan, rounded);
  return ma::TruncIOp::create(builder, dst_ty, selected);
}

class FloatStorageTypeConverter : public mlir::TypeConverter {
 public:
  explicit FloatStorageTypeConverter(mlir::MLIRContext* /*context*/) {
    addConversion([](mlir::Type type) { return type; });

    // This mirrors LLVMTypeConverter's unsupported-low-precision-float rule:
    // represent the value with an integer of the same bit width. Metal also
    // applies this to BF16 because MSL does not expose BF16 as a scalar type.
    addConversion(
        [](mlir::FloatType type) { return ConvertFloatTypeToStorage(type); });

    addConversion([this](mlir::VectorType type) -> mlir::Type {
      mlir::Type element_type = convertType(type.getElementType());
      if (!element_type) {
        return {};
      }
      if (element_type == type.getElementType()) {
        return type;
      }
      return type.clone(element_type);
    });

    addConversion([this](mlir::RankedTensorType type) -> mlir::Type {
      mlir::Type element_type = convertType(type.getElementType());
      if (!element_type) {
        return {};
      }
      if (element_type == type.getElementType()) {
        return type;
      }
      return type.clone(element_type);
    });

    addConversion([this](mlir::FunctionType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> inputs;
      llvm::SmallVector<mlir::Type> results;
      if (failed(convertTypes(type.getInputs(), inputs)) ||
          failed(convertTypes(type.getResults(), results))) {
        return {};
      }
      return mlir::FunctionType::get(type.getContext(), inputs, results);
    });
  }
};

struct ConvertConstantOp : public mlir::OpConversionPattern<ma::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ConstantOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasLoweredFloatElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not a lowered float constant");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::TypedAttr converted_attr =
        ConvertFloatAttrToStorage(op.getValue(), converted_type);
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
    if (!HasLoweredFloatElementType(op.getIn().getType()) &&
        !HasLoweredFloatElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "no lowered float bitcast operand");
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

struct ConvertExtFOp : public mlir::OpConversionPattern<ma::ExtFOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasBf16ElementType(op.getIn().getType()) ||
        !mlir::isa<mlir::Float32Type>(
            mlir::getElementTypeOrSelf(op.getType()))) {
      return rewriter.notifyMatchFailure(op, "not bf16 -> f32");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    rewriter.replaceOp(
        op, EmitBF16BitsToF32(op.getType(), adaptor.getIn(), builder));
    return mlir::success();
  }
};

struct ConvertTruncFOp : public mlir::OpConversionPattern<ma::TruncFOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!mlir::isa<mlir::Float32Type>(
            mlir::getElementTypeOrSelf(op.getIn().getType())) ||
        !HasBf16ElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not f32 -> bf16");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    rewriter.replaceOp(
        op, EmitF32ToBF16Bits(converted_type, adaptor.getIn(), builder));
    return mlir::success();
  }
};

struct ConvertTensorExtract
    : public mlir::OpConversionPattern<mlir::tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
        op, converted_type, adaptor.getTensor(), adaptor.getIndices());
    return mlir::success();
  }
};

struct ConvertTensorInsert
    : public mlir::OpConversionPattern<mlir::tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto new_op = mlir::tensor::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getScalar(), adaptor.getDest(),
        adaptor.getIndices());
    rewriter.replaceOp(op, new_op.getResult());
    return mlir::success();
  }
};

struct ConvertSelectOp : public mlir::OpConversionPattern<ma::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::SelectOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasLoweredFloatElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "result is not a lowered float type");
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

struct ConvertVectorExtract
    : public mlir::OpConversionPattern<mlir::vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasLoweredFloatElementType(op.getSource().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "source is not a lowered float type");
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
    if (!HasLoweredFloatElementType(op.getDest().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "dest is not a lowered float type");
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
    if (!HasLoweredFloatElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "result is not a lowered float type");
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

struct ConvertVectorTransferRead
    : public mlir::OpConversionPattern<mlir::vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::TransferReadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    auto converted_vector_type =
        mlir::dyn_cast_if_present<mlir::VectorType>(converted_type);
    if (!converted_vector_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    rewriter.replaceOpWithNewOp<mlir::vector::TransferReadOp>(
        op, converted_vector_type, adaptor.getBase(), adaptor.getIndices(),
        op.getPermutationMapAttr(), adaptor.getPadding(), adaptor.getMask(),
        op.getInBoundsAttr());
    return mlir::success();
  }
};

struct ConvertVectorTransferWrite
    : public mlir::OpConversionPattern<mlir::vector::TransferWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::TransferWriteOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> result_types;
    if (op->getNumResults() != 0) {
      mlir::Type converted_type =
          getTypeConverter()->convertType(op.getResult().getType());
      if (!converted_type) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }
      result_types.push_back(converted_type);
    }

    auto new_op = mlir::vector::TransferWriteOp::create(
        rewriter, op.getLoc(), mlir::TypeRange(result_types),
        adaptor.getValueToStore(), adaptor.getBase(), adaptor.getIndices(),
        op.getPermutationMapAttr(), adaptor.getMask(), op.getInBoundsAttr());
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
  }
};

struct ConvertPoisonOp : public mlir::OpConversionPattern<mlir::ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ub::PoisonOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasLoweredFloatElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not a lowered float poison");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, converted_type);
    return mlir::success();
  }
};

class LowerFloatStoragePass
    : public impl::LowerFloatStoragePassBase<LowerFloatStoragePass> {
 public:
  using LowerFloatStoragePassBase::LowerFloatStoragePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    FloatStorageTypeConverter converter(context);

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertBitcastOp, ConvertConstantOp, ConvertExtFOp,
                 ConvertPoisonOp, ConvertSelectOp, ConvertTruncFOp,
                 ConvertTensorExtract, ConvertTensorInsert,
                 ConvertVectorBroadcast, ConvertVectorExtract,
                 ConvertVectorFromElements, ConvertVectorInsert,
                 ConvertVectorTransferRead, ConvertVectorTransferWrite>(
        converter, context);
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

std::unique_ptr<mlir::Pass> CreateLowerFloatStoragePass() {
  return std::make_unique<LowerFloatStoragePass>();
}

}  // namespace metal
}  // namespace xla
