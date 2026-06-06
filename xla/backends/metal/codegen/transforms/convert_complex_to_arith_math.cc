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
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/metal/codegen/transforms/passes.h"

namespace xla {
namespace metal {

namespace ma = ::mlir::arith;

#define GEN_PASS_DEF_CONVERTCOMPLEXTOARITHMATHPASS
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

namespace {

constexpr int64_t kRealPos = 0;
constexpr int64_t kImagPos = 1;

mlir::VectorType GetConvertedComplexType(mlir::ComplexType type) {
  return mlir::VectorType::get({2}, type.getElementType());
}

mlir::Type ConvertComplexTensorType(mlir::RankedTensorType type) {
  auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type.getElementType());
  if (!complex_type) {
    return type;
  }
  if (!type.hasStaticShape()) {
    return {};
  }
  return mlir::RankedTensorType::get({type.getNumElements() * 2},
                                     complex_type.getElementType(),
                                     type.getEncoding());
}

bool HasComplexElementType(mlir::Type type) {
  return mlir::isa<mlir::ComplexType>(mlir::getElementTypeOrSelf(type));
}

mlir::Value ExtractReal(mlir::Location loc, mlir::Value complex_value,
                        mlir::PatternRewriter& rewriter) {
  return mlir::vector::ExtractOp::create(rewriter, loc, complex_value,
                                         kRealPos);
}

mlir::Value ExtractImag(mlir::Location loc, mlir::Value complex_value,
                        mlir::PatternRewriter& rewriter) {
  return mlir::vector::ExtractOp::create(rewriter, loc, complex_value,
                                         kImagPos);
}

mlir::Value BuildComplexVector(mlir::Location loc, mlir::Type result_type,
                               mlir::Value real, mlir::Value imag,
                               mlir::PatternRewriter& rewriter) {
  return mlir::vector::FromElementsOp::create(rewriter, loc, result_type,
                                              mlir::ValueRange{real, imag});
}

mlir::Value GetLinearIndex(mlir::ValueRange indices,
                           mlir::ImplicitLocOpBuilder& builder) {
  if (indices.empty()) return ma::ConstantIndexOp::create(builder, 0);
  return indices.front();
}

std::pair<mlir::Value, mlir::Value> GetComplexComponentIndices(
    mlir::Value linear_index, mlir::ImplicitLocOpBuilder& builder) {
  mlir::Value two = ma::ConstantIndexOp::create(builder, 2);
  mlir::Value real_index = ma::MulIOp::create(builder, linear_index, two);
  mlir::Value imag_index = ma::AddIOp::create(
      builder, real_index, ma::ConstantIndexOp::create(builder, 1));
  return {real_index, imag_index};
}

class ComplexToArithMathTypeConverter : public mlir::TypeConverter {
 public:
  ComplexToArithMathTypeConverter() {
    addConversion([](mlir::Type type) { return type; });

    addConversion([](mlir::ComplexType type) -> mlir::Type {
      return GetConvertedComplexType(type);
    });

    addConversion([](mlir::RankedTensorType type) -> mlir::Type {
      return ConvertComplexTensorType(type);
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

struct ConstantOpLowering final
    : public mlir::OpConversionPattern<mlir::complex::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::complex::ConstantOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto dst_type = mlir::cast<mlir::VectorType>(
        getTypeConverter()->convertType(op.getType()));

    auto value = mlir::dyn_cast<mlir::ArrayAttr>(op.getValue());
    if (!value || value.size() != 2) {
      return rewriter.notifyMatchFailure(op, "expected two complex components");
    }

    auto real_attr = mlir::dyn_cast<mlir::TypedAttr>(value[0]);
    auto imag_attr = mlir::dyn_cast<mlir::TypedAttr>(value[1]);
    if (!real_attr || !imag_attr) {
      return rewriter.notifyMatchFailure(op, "expected typed component attrs");
    }

    mlir::Location loc = op.getLoc();
    mlir::Value real =
        ma::ConstantOp::create(rewriter, loc, real_attr.getType(), real_attr);
    mlir::Value imag =
        ma::ConstantOp::create(rewriter, loc, imag_attr.getType(), imag_attr);
    rewriter.replaceOp(op,
                       BuildComplexVector(loc, dst_type, real, imag, rewriter));
    return mlir::success();
  }
};

struct CreateOpConversion final
    : public mlir::OpConversionPattern<mlir::complex::CreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::complex::CreateOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto dst_type = mlir::cast<mlir::VectorType>(
        getTypeConverter()->convertType(op.getType()));
    rewriter.replaceOp(
        op, BuildComplexVector(op.getLoc(), dst_type, adaptor.getReal(),
                               adaptor.getImaginary(), rewriter));
    return mlir::success();
  }
};

struct ReOpConversion final
    : public mlir::OpConversionPattern<mlir::complex::ReOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::complex::ReOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(
        op, ExtractReal(op.getLoc(), adaptor.getComplex(), rewriter));
    return mlir::success();
  }
};

struct ImOpConversion final
    : public mlir::OpConversionPattern<mlir::complex::ImOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::complex::ImOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(
        op, ExtractImag(op.getLoc(), adaptor.getComplex(), rewriter));
    return mlir::success();
  }
};

struct SelectOpConversion final
    : public mlir::OpConversionPattern<ma::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::SelectOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!mlir::isa<mlir::ComplexType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not complex");
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

struct PoisonOpConversion final
    : public mlir::OpConversionPattern<mlir::ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ub::PoisonOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasComplexElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not complex");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, converted_type);
    return mlir::success();
  }
};

struct TensorExtractConversion final
    : public mlir::OpConversionPattern<mlir::tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!mlir::isa<mlir::ComplexType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not complex");
    }
    if (op.getIndices().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "only rank-0 and rank-1 tensors are supported");
    }

    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value linear_index = GetLinearIndex(adaptor.getIndices(), builder);
    auto [real_index, imag_index] =
        GetComplexComponentIndices(linear_index, builder);
    mlir::Value real = mlir::tensor::ExtractOp::create(
        builder, adaptor.getTensor(), real_index);
    mlir::Value imag = mlir::tensor::ExtractOp::create(
        builder, adaptor.getTensor(), imag_index);
    rewriter.replaceOp(op, BuildComplexVector(op.getLoc(), converted_type, real,
                                              imag, rewriter));
    return mlir::success();
  }
};

struct TensorInsertConversion final
    : public mlir::OpConversionPattern<mlir::tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!mlir::isa<mlir::ComplexType>(op.getScalar().getType())) {
      return rewriter.notifyMatchFailure(op, "scalar is not complex");
    }
    if (op.getIndices().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "only rank-0 and rank-1 tensors are supported");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    mlir::Value linear_index = GetLinearIndex(adaptor.getIndices(), builder);
    auto [real_index, imag_index] =
        GetComplexComponentIndices(linear_index, builder);

    mlir::Value real = ExtractReal(op.getLoc(), adaptor.getScalar(), rewriter);
    mlir::Value imag = ExtractImag(op.getLoc(), adaptor.getScalar(), rewriter);
    mlir::Value with_real = mlir::tensor::InsertOp::create(
        builder, real, adaptor.getDest(), real_index);
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertOp>(op, imag, with_real,
                                                        imag_index);
    return mlir::success();
  }
};

struct AllocateSharedConversion final
    : public mlir::OpConversionPattern<::xla::gpu::AllocateSharedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::gpu::AllocateSharedOp op, OpAdaptor /*adaptor*/,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!HasComplexElementType(op.getType())) {
      return rewriter.notifyMatchFailure(op, "result is not complex");
    }
    mlir::Type converted_type = getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<::xla::gpu::AllocateSharedOp>(op,
                                                              converted_type);
    return mlir::success();
  }
};

struct SyncThreadsConversion final
    : public mlir::OpConversionPattern<::xla::gpu::SyncThreadsOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::gpu::SyncThreadsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    // SyncThreadsOp has TypesMatchWith between operands and results, so each
    // result type tracks its operand. Rewrite both via the converter.
    llvm::SmallVector<mlir::Type> new_result_types;
    new_result_types.reserve(op.getNumResults());
    bool any_complex = false;
    for (mlir::Type t : op.getResultTypes()) {
      mlir::Type conv = getTypeConverter()->convertType(t);
      if (!conv) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }
      if (HasComplexElementType(t)) any_complex = true;
      new_result_types.push_back(conv);
    }
    if (!any_complex) {
      return rewriter.notifyMatchFailure(op, "no complex operands");
    }
    rewriter.replaceOpWithNewOp<::xla::gpu::SyncThreadsOp>(
        op, new_result_types, adaptor.getOperands());
    return mlir::success();
  }
};

class ConvertComplexToArithMathPass
    : public impl::ConvertComplexToArithMathPassBase<
          ConvertComplexToArithMathPass> {
 public:
  using ConvertComplexToArithMathPassBase::ConvertComplexToArithMathPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    ComplexToArithMathTypeConverter converter;

    mlir::RewritePatternSet patterns(context);
    patterns.add<AllocateSharedConversion, ConstantOpLowering,
                 CreateOpConversion, ImOpConversion, PoisonOpConversion,
                 ReOpConversion, SelectOpConversion, SyncThreadsConversion,
                 TensorExtractConversion, TensorInsertConversion>(converter,
                                                                  context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);
    // Rewrite func.call sites whose callee signatures have been rewritten
    // (complex<f32> → vector<2xf32>). Without this the call's operand/result
    // types stay complex<f32> while the callee body is converted, and full
    // conversion fails with "failed to legalize operation 'func.call'".
    mlir::populateCallOpTypeConversionPattern(patterns, converter);

    mlir::ConversionTarget target(*context);
    target.addIllegalDialect<mlir::complex::ComplexDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) {
          return converter.isLegal(op.getOperandTypes()) &&
                 converter.isLegal(op.getResultTypes());
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

std::unique_ptr<mlir::Pass> CreateConvertComplexToArithMathPass() {
  return std::make_unique<ConvertComplexToArithMathPass>();
}

}  // namespace metal
}  // namespace xla
