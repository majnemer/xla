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

#include "xla/backends/metal/codegen/msl_kernel_emitter.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace metal {
namespace {

std::unique_ptr<mlir::MLIRContext> MakeMlirContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect>();
  return std::make_unique<mlir::MLIRContext>(registry);
}

TEST(MslKernelEmitter, RunsCsePassBeforeTranslation) {
  auto ctx = MakeMlirContext();
  // Two arith.constant ops with identical attributes; CSE should fold them
  // into a single value before TranslateToMSL runs. Without CSE the
  // emitter would produce two `static_cast<int>(7)` lines.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @two_constants(%out: tensor<2xi32> {xla.slice_index = 0 : i64})
          -> tensor<2xi32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %k_a = arith.constant 7 : i32
        %k_b = arith.constant 7 : i32
        %t0 = tensor.insert %k_a into %out[%c0] : tensor<2xi32>
        %t1 = tensor.insert %k_b into %t0[%c1] : tensor<2xi32>
        return %t1 : tensor<2xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, EmitMslKernel(*module));

  // After CSE only one `static_cast<int>(7)` literal should remain.
  size_t first = result.source().find("static_cast<int>(7)");
  ASSERT_NE(first, std::string::npos) << result.source();
  size_t second = result.source().find("static_cast<int>(7)", first + 1);
  EXPECT_EQ(second, std::string::npos)
      << "CSE did not eliminate the duplicate constant; emitter still "
         "produced two int 7 literals:\n"
      << result.source();
}

TEST(MslKernelEmitter, ExpandsLog1pBeforeTranslation) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @log1p_kernel(%a: tensor<1xf32> {xla.slice_index = 0 : i64},
                              %out: tensor<1xf32> {xla.slice_index = 1 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %x = tensor.extract %a[%c0] : tensor<1xf32>
        %y = math.log1p %x : f32
        %r = tensor.insert %y into %out[%c0] : tensor<1xf32>
        return %r : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, EmitMslKernel(*module));
  EXPECT_EQ(result.source().find("__xla_log1p"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("metal::log("), std::string::npos)
      << result.source();
}

TEST(MslKernelEmitter, EmitsSingleElementVectorTransfersAsScalars) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @v1(%a: tensor<1xi64> {xla.slice_index = 0 : i64},
                    %b: tensor<1xi64> {xla.slice_index = 1 : i64})
          -> tensor<1xi64> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %pad = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]}
            : tensor<1xi64>, vector<1xi64>
        %x = vector.extract %v[0] : i64 from vector<1xi64>
        %y = arith.addi %x, %one : i64
        %out = vector.from_elements %y : vector<1xi64>
        %r = vector.transfer_write %out, %b[%c0] {in_bounds = [true]}
            : vector<1xi64>, tensor<1xi64>
        return %r : tensor<1xi64>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, EmitMslKernel(*module));
  EXPECT_EQ(result.source().find("long1"), std::string::npos)
      << result.source();
  EXPECT_EQ(result.source().find("int2"), std::string::npos) << result.source();
  EXPECT_NE(result.source().find("const device long*"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("device long*"), std::string::npos)
      << result.source();
}

TEST(MslKernelEmitter, PassesThroughKernelSignatureUnchanged) {
  auto ctx = MakeMlirContext();
  // CSE and canonicalize must not strip the entry-point function or its
  // xla.entry / xla.slice_index attributes.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @passthrough(%a: tensor<4xf32> {xla.slice_index = 0 : i64},
                              %b: tensor<4xf32> {xla.slice_index = 1 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %x = tensor.extract %a[%c0] : tensor<4xf32>
        %r = tensor.insert %x into %b[%c0] : tensor<4xf32>
        return %r : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, EmitMslKernel(*module));
  EXPECT_EQ(result.entry_point(), "passthrough");
  EXPECT_NE(result.source().find("device float* arg0 [[buffer(0)]]"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("device float* arg1 [[buffer(1)]]"),
            std::string::npos)
      << result.source();
}

}  // namespace
}  // namespace metal
}  // namespace xla
