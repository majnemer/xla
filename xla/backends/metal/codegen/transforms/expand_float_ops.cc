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

#include <array>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/metal/codegen/transforms/passes.h"

namespace xla {
namespace metal {

namespace ma = ::mlir::arith;

#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

namespace {

mlir::Value GetFloatConst(mlir::ImplicitLocOpBuilder& builder,
                          mlir::Type type, double value) {
  mlir::Type element_type = mlir::getElementTypeOrSelf(type);
  mlir::TypedAttr attr = builder.getFloatAttr(element_type, value);
  if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type)) {
    attr = mlir::SplatElementsAttr::get(shaped_type, attr);
  }
  return ma::ConstantOp::create(builder, type, attr);
}

mlir::Value Add(mlir::ImplicitLocOpBuilder& builder, mlir::Value lhs,
                mlir::Value rhs, ma::FastMathFlagsAttr fastmath) {
  return ma::AddFOp::create(builder, lhs, rhs, fastmath);
}

mlir::Value Mul(mlir::ImplicitLocOpBuilder& builder, mlir::Value lhs,
                mlir::Value rhs, ma::FastMathFlagsAttr fastmath) {
  return ma::MulFOp::create(builder, lhs, rhs, fastmath);
}

// MSL has no log1p intrinsic. Expand to the same Cephes-style approximation
// that used to be emitted as a translator-side helper:
//   |x| < sqrt(2)-1: x - 0.5*x^2 + x^3*N(x)/D(x)
//   otherwise:       log(x + 1)
class RewriteLog1pPattern : public mlir::OpRewritePattern<mlir::math::Log1pOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Log1pOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();
    mlir::Type element_type = mlir::getElementTypeOrSelf(type);
    if (!mlir::isa<mlir::Float16Type, mlir::Float32Type>(element_type)) {
      return rewriter.notifyMatchFailure(op, "not an f16/f32 log1p");
    }

    static constexpr std::array<double, 7> kD = {
        1.0,
        1.5062909083469192043167e1,
        8.3047565967967209469434e1,
        2.2176239823732856465394e2,
        3.0909872225312059774938e2,
        2.1642788614495947685003e2,
        6.0118660497603843919306e1,
    };
    static constexpr std::array<double, 7> kN = {
        4.5270000862445199635215e-5,
        4.9854102823193375972212e-1,
        6.5787325942061044846969e0,
        2.9911919328553073277375e1,
        6.0949667980987787057556e1,
        5.7112963590585538103336e1,
        2.0039553499201281259648e1,
    };

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    ma::FastMathFlagsAttr fastmath = op.getFastmathAttr();
    auto c = [&](double value) -> mlir::Value {
      return GetFloatConst(builder, type, value);
    };

    mlir::Value x = op.getOperand();
    mlir::Value d = c(0.0);
    mlir::Value n = c(0.0);
    for (int i = 0; i < 7; ++i) {
      d = Add(builder, Mul(builder, d, x, fastmath), c(kD[i]), fastmath);
      n = Add(builder, Mul(builder, n, x, fastmath), c(kN[i]), fastmath);
    }

    mlir::Value xsq = Mul(builder, x, x, fastmath);
    mlir::Value x_plus_one = Add(builder, x, c(1.0), fastmath);
    mlir::Value for_large =
        mlir::math::LogOp::create(builder, x_plus_one, fastmath);

    mlir::Value cubic = Mul(builder, x, xsq, fastmath);
    mlir::Value rational = ma::DivFOp::create(builder, n, d, fastmath);
    mlir::Value correction = Mul(builder, cubic, rational, fastmath);
    mlir::Value quadratic = Mul(builder, c(-0.5), xsq, fastmath);
    mlir::Value for_small =
        Add(builder, Add(builder, x, quadratic, fastmath), correction,
            fastmath);

    mlir::Value abs_x = mlir::math::AbsFOp::create(builder, x, fastmath);
    mlir::Value use_small = ma::CmpFOp::create(
        builder, ma::CmpFPredicate::OLT, abs_x, c(0.41421356237309504880));

    rewriter.replaceOpWithNewOp<ma::SelectOp>(op, use_small, for_small,
                                              for_large);
    return mlir::success();
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteLog1pPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass() {
  return std::make_unique<ExpandFloatOpsPass>();
}

}  // namespace metal
}  // namespace xla
