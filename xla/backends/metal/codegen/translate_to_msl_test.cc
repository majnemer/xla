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

#include "xla/backends/metal/codegen/translate_to_msl.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/service/name_uniquer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace metal {
namespace {

std::unique_ptr<mlir::MLIRContext> MakeMlirContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect, ::xla::gpu::XlaGpuDialect,
                  ::xla::XlaDialect>();
  return std::make_unique<mlir::MLIRContext>(registry);
}

TEST(TranslateToMSL, EmitsKernelSignatureForFloatBuffers) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @add(%a: tensor<10xf32> {xla.slice_index = 0 : i64},
                     %b: tensor<10xf32> {xla.slice_index = 1 : i64},
                     %out: tensor<10xf32> {xla.slice_index = 2 : i64})
          -> tensor<10xf32> attributes {xla.entry} {
        return %out : tensor<10xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.entry_point(), "add");
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void add("
            "\n    device float* arg0 [[buffer(0)]],"
            "\n    device float* arg1 [[buffer(1)]],"
            "\n    device float* arg2 [[buffer(2)]]) {\n}\n");
}

TEST(TranslateToMSL, RecoversIntegerTypesAndSigning) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @mixed(%a: tensor<4xi32> {xla.slice_index = 0 : i64},
                       %b: tensor<4xui32> {xla.slice_index = 1 : i64},
                       %c: tensor<4xi8> {xla.slice_index = 2 : i64},
                       %d: tensor<4xf16> {xla.slice_index = 3 : i64})
          -> tensor<4xi32> attributes {xla.entry} {
        return %a : tensor<4xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void mixed("
            "\n    device int* arg0 [[buffer(0)]],"
            "\n    device uint* arg1 [[buffer(1)]],"
            "\n    device char* arg2 [[buffer(2)]],"
            "\n    device half* arg3 [[buffer(3)]]) {\n}\n");
}

TEST(TranslateToMSL, PicksEntryFuncAmongMultipleFuncs) {
  // The emitter pipeline typically produces multiple funcs per module — one
  // entry plus helpers for partitioned subgraphs. Only the xla.entry func
  // should be picked for the kernel signature.
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func private @helper(%x: f32) -> f32 {
        return %x : f32
      }
      func.func @entry_kernel(%a: tensor<8xf32> {xla.slice_index = 0 : i64})
          -> tensor<8xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %x = tensor.extract %a[%c0] : tensor<8xf32>
        %y = func.call @helper(%x) : (f32) -> f32
        %out = tensor.insert %y into %a[%c0] : tensor<8xf32>
        return %out : tensor<8xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.entry_point(), "entry_kernel");
  EXPECT_NE(result.source().find("kernel void entry_kernel"),
            std::string::npos);
  EXPECT_NE(result.source().find("float entry_kernel_helper(float"),
            std::string::npos);
  EXPECT_NE(result.source().find("entry_kernel_helper("),
            std::string::npos);
  EXPECT_EQ(result.source().find("float helper(float"), std::string::npos);
}

TEST(TranslateToMSL, EmitsDeviceFunctionWithMultipleResults) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func private @split(%x: i32) -> (i32, i32) {
        %one = arith.constant 1 : i32
        %y = arith.addi %x, %one : i32
        return %x, %y : i32, i32
      }
      func.func @multi_return(
          %src: tensor<1xi32> {xla.slice_index = 0 : i64},
          %dst0: tensor<1xi32> {xla.slice_index = 1 : i64},
          %dst1: tensor<1xi32> {xla.slice_index = 2 : i64})
          -> (tensor<1xi32>, tensor<1xi32>) attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %src[%i] : tensor<1xi32>
        %y0, %y1 = func.call @split(%x) : (i32) -> (i32, i32)
        %out0 = tensor.insert %y0 into %dst0[%i] : tensor<1xi32>
        %out1 = tensor.insert %y1 into %dst1[%i] : tensor<1xi32>
        return %out0, %out1 : tensor<1xi32>, tensor<1xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("struct multi_return_split_result"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("int result0;"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("int result1;"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find(
                "multi_return_split_result multi_return_split(int"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find(
                "return multi_return_split_result{arg0,"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find(".result0"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find(".result1"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, MangledHelperNamesDoNotAliasAcrossEntries) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kFooInput = R"mlir(
    module {
      func.func private @bar_baz(%x: f32) -> f32 {
        return %x : f32
      }
      func.func @foo(%a: tensor<8xf32> {xla.slice_index = 0 : i64})
          -> tensor<8xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %x = tensor.extract %a[%c0] : tensor<8xf32>
        %y = func.call @bar_baz(%x) : (f32) -> f32
        %out = tensor.insert %y into %a[%c0] : tensor<8xf32>
        return %out : tensor<8xf32>
      }
    }
  )mlir";
  constexpr absl::string_view kFooBarInput = R"mlir(
    module {
      func.func private @baz(%x: f32) -> f32 {
        return %x : f32
      }
      func.func @foo_bar(%a: tensor<8xf32> {xla.slice_index = 0 : i64})
          -> tensor<8xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %x = tensor.extract %a[%c0] : tensor<8xf32>
        %y = func.call @baz(%x) : (f32) -> f32
        %out = tensor.insert %y into %a[%c0] : tensor<8xf32>
        return %out : tensor<8xf32>
      }
    }
  )mlir";
  auto foo_module =
      mlir::parseSourceString<mlir::ModuleOp>(kFooInput, ctx.get());
  ASSERT_TRUE(foo_module);
  auto foo_bar_module =
      mlir::parseSourceString<mlir::ModuleOp>(kFooBarInput, ctx.get());
  ASSERT_TRUE(foo_bar_module);

  NameUniquer function_name_uniquer;
  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource foo_result,
                          TranslateToMSL(*foo_module, &function_name_uniquer));
  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource foo_bar_result,
                          TranslateToMSL(*foo_bar_module,
                                         &function_name_uniquer));

  constexpr absl::string_view kFooBarBazFromFoo = "foo_bar_baz";
  constexpr absl::string_view kFooBarBazFromFooBar = "foo_bar_baz__1";
  EXPECT_NE(foo_result.source().find(
                absl::StrCat("float ", kFooBarBazFromFoo, "(float")),
            std::string::npos);
  EXPECT_EQ(foo_result.source().find(kFooBarBazFromFooBar), std::string::npos);
  EXPECT_NE(foo_bar_result.source().find(
                absl::StrCat("float ", kFooBarBazFromFooBar, "(float")),
            std::string::npos);
  EXPECT_EQ(foo_bar_result.source().find(
                absl::StrCat(kFooBarBazFromFoo, "(")),
            std::string::npos);
}

TEST(TranslateToMSL, OmitsUnusedGpuShuffleValidResult) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @shuffle_value_only(
          %src: tensor<1xf32> {xla.slice_index = 0 : i64},
          %dst: tensor<1xf32> {xla.slice_index = 1 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %offset = arith.constant 1 : i32
        %width = arith.constant 32 : i32
        %x = tensor.extract %src[%i] : tensor<1xf32>
        %shuf, %valid = gpu.shuffle down %x, %offset, %width : f32
        %out = tensor.insert %shuf into %dst[%i] : tensor<1xf32>
        return %out : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("metal::simd_shuffle_down"),
            std::string::npos);
  EXPECT_EQ(result.source().find("bool v"), std::string::npos);
}

TEST(TranslateToMSL, EmitsUsedGpuShuffleValidResult) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @shuffle_valid_used(
          %src: tensor<1xf32> {xla.slice_index = 0 : i64},
          %dst: tensor<1xi1> {xla.slice_index = 1 : i64})
          -> tensor<1xi1> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %offset = arith.constant 1 : i32
        %width = arith.constant 32 : i32
        %x = tensor.extract %src[%i] : tensor<1xf32>
        %shuf, %valid = gpu.shuffle down %x, %offset, %width : f32
        %out = tensor.insert %valid into %dst[%i] : tensor<1xi1>
        return %out : tensor<1xi1>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("metal::simd_shuffle_down"),
            std::string::npos);
  EXPECT_NE(result.source().find("bool v"), std::string::npos);
  EXPECT_NE(result.source().find(" = true;"), std::string::npos);
}

TEST(TranslateToMSL, EmitsSingleElementVectorBitcastAsScalar) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @split_join_u64(
          %src: tensor<1xi64> {xla.slice_index = 0 : i64},
          %dst: tensor<1xi64> {xla.slice_index = 1 : i64})
          -> tensor<1xi64> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %src[%i] : tensor<1xi64>
        %v1 = vector.broadcast %x : i64 to vector<1xi64>
        %v2 = vector.bitcast %v1 : vector<1xi64> to vector<2xi32>
        %lo = vector.extract %v2[0] : i32 from vector<2xi32>
        %hi = vector.extract %v2[1] : i32 from vector<2xi32>
        %v3 = vector.from_elements %lo, %hi : vector<2xi32>
        %v4 = vector.bitcast %v3 : vector<2xi32> to vector<1xi64>
        %y = vector.extract %v4[0] : i64 from vector<1xi64>
        %out = tensor.insert %y into %dst[%i] : tensor<1xi64>
        return %out : tensor<1xi64>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source().find("long1"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("int2"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("as_type<int2>"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("as_type<long>"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, RejectsMissingEntryAttribute) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @noentry(%a: tensor<4xf32> {xla.slice_index = 0 : i64})
          -> tensor<4xf32> {
        return %a : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  auto status = TranslateToMSL(*module).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("xla.entry"), absl::string_view::npos);
}

TEST(TranslateToMSL, RejectsMissingSliceIndex) {
  auto ctx = MakeMlirContext();
  // The function has xla.entry but the second arg is missing xla.slice_index.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @missing_slice(%a: tensor<4xf32> {xla.slice_index = 0 : i64},
                                %b: tensor<4xf32>)
          -> tensor<4xf32> attributes {xla.entry} {
        return %a : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  auto status = TranslateToMSL(*module).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("xla.slice_index"), absl::string_view::npos);
}

TEST(TranslateToMSL, EmitsBodyForCopyKernel) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @copy(%src: tensor<4xf32> {xla.slice_index = 0 : i64},
                      %dst: tensor<4xf32> {xla.slice_index = 1 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %src[%i] : tensor<4xf32>
        %out = tensor.insert %x into %dst[%i] : tensor<4xf32>
        return %out : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void copy("
            "\n    device float* arg0 [[buffer(0)]],"
            "\n    device float* arg1 [[buffer(1)]]) {\n"
            "  long v0 = static_cast<long>(0);\n"
            "  float v1 = arg0[v0];\n"
            "  arg1[v0] = v1;\n"
            "}\n");
}

TEST(TranslateToMSL, EmitsBodyForArithChain) {
  auto ctx = MakeMlirContext();
  // c[0] = (a[0] + 1.0) * 2.0
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @scale(%a: tensor<4xf32> {xla.slice_index = 0 : i64},
                       %c: tensor<4xf32> {xla.slice_index = 1 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<4xf32>
        %k1 = arith.constant 1.000000e+00 : f32
        %y = arith.addf %x, %k1 : f32
        %k2 = arith.constant 2.000000e+00 : f32
        %z = arith.mulf %y, %k2 : f32
        %out = tensor.insert %z into %c[%i] : tensor<4xf32>
        return %out : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void scale("
            "\n    device float* arg0 [[buffer(0)]],"
            "\n    device float* arg1 [[buffer(1)]]) {\n"
            "  long v0 = static_cast<long>(0);\n"
            "  float v1 = arg0[v0];\n"
            "  float v2 = 1.00000000f;\n"
            "  float v3 = v1 + v2;\n"
            "  float v4 = 2.00000000f;\n"
            "  float v5 = v3 * v4;\n"
            "  arg1[v0] = v5;\n"
            "}\n");
}

TEST(TranslateToMSL, EmitsAtomicRMWStoreAsCasLoop) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @atomic_store(%dst: tensor<4xi32> {xla.slice_index = 0 : i64},
                              %updates: tensor<4xi32> {xla.slice_index = 1 : i64})
          -> tensor<4xi32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %u = tensor.extract %updates[%i] : tensor<4xi32>
        %out = xla.atomic_rmw %dst[%i] : tensor<4xi32> {
          ^bb0(%current : i32):
            xla.yield %u : i32
        }
        return %out : tensor<4xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void atomic_store("
            "\n    device int* arg0 [[buffer(0)]],"
            "\n    device int* arg1 [[buffer(1)]]) {\n"
            "  long v0 = static_cast<long>(0);\n"
            "  int v1 = arg1[v0];\n"
            "  device atomic_int* v2 = (device atomic_int*)(&arg0[v0]);\n"
            "  int v3 = atomic_load_explicit(v2, memory_order_relaxed);\n"
            "  bool v4 = false;\n"
            "  do {\n"
            "    int v5 = static_cast<int>(v3);\n"
            "    int v6 = static_cast<int>(v1);\n"
            "    v4 = atomic_compare_exchange_weak_explicit(v2, &v3, v6, memory_order_relaxed, memory_order_relaxed);\n"
            "  } while (!v4);\n"
            "}\n");
}

TEST(TranslateToMSL, EmitsAtomicRMWF32AsCasLoop) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @atomic_add(%dst: tensor<4xf32> {xla.slice_index = 0 : i64},
                            %updates: tensor<4xf32> {xla.slice_index = 1 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %u = tensor.extract %updates[%i] : tensor<4xf32>
        %out = xla.atomic_rmw %dst[%i] : tensor<4xf32> {
          ^bb0(%current : f32):
            %sum = arith.addf %current, %u : f32
            xla.yield %sum : f32
        }
        return %out : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void atomic_add("
            "\n    device float* arg0 [[buffer(0)]],"
            "\n    device float* arg1 [[buffer(1)]]) {\n"
            "  long v0 = static_cast<long>(0);\n"
            "  float v1 = arg1[v0];\n"
            "  device atomic_uint* v2 = (device atomic_uint*)(&arg0[v0]);\n"
            "  uint v3 = atomic_load_explicit(v2, memory_order_relaxed);\n"
            "  bool v4 = false;\n"
            "  do {\n"
            "    float v5 = as_type<float>(v3);\n"
            "    float v6 = v5 + v1;\n"
            "    uint v7 = as_type<uint>(v6);\n"
            "    v4 = atomic_compare_exchange_weak_explicit(v2, &v3, v7, memory_order_relaxed, memory_order_relaxed);\n"
            "  } while (!v4);\n"
            "}\n");
}

TEST(TranslateToMSL, EmitsAtomicRMWF16AsWordCasLoop) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @atomic_add_half(
          %dst: tensor<4xf16> {xla.slice_index = 0 : i64},
          %updates: tensor<4xf16> {xla.slice_index = 1 : i64})
          -> tensor<4xf16> attributes {xla.entry} {
        %i = arith.constant 1 : index
        %u = tensor.extract %updates[%i] : tensor<4xf16>
        %out = xla.atomic_rmw %dst[%i] : tensor<4xf16> {
          ^bb0(%current : f16):
            %sum = arith.addf %current, %u : f16
            xla.yield %sum : f16
        }
        return %out : tensor<4xf16>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("device atomic_uint*"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("ushort"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("as_type<half>"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("as_type<ushort>"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("atomic_compare_exchange_weak_explicit"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsAtomicRMWI8AsWordCasLoop) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @atomic_i8(%dst: tensor<4xi8> {xla.slice_index = 0 : i64},
                           %updates: tensor<4xi8> {xla.slice_index = 1 : i64})
          -> tensor<4xi8> attributes {xla.entry} {
        %i = arith.constant 1 : index
        %u = tensor.extract %updates[%i] : tensor<4xi8>
        %out = xla.atomic_rmw %dst[%i] : tensor<4xi8> {
          ^bb0(%current : i8):
            %mask = arith.constant 15 : i8
            %preserved = arith.andi %current, %mask : i8
            %new = arith.ori %preserved, %u : i8
            xla.yield %new : i8
        }
        return %out : tensor<4xi8>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source(),
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void atomic_i8("
            "\n    device char* arg0 [[buffer(0)]],"
            "\n    device char* arg1 [[buffer(1)]]) {\n"
            "  long v0 = static_cast<long>(1);\n"
            "  char v1 = arg1[v0];\n"
            "  ulong v2 = static_cast<ulong>(v0);\n"
            "  ulong v3 = v2;\n"
            "  device atomic_uint* v4 = ((device atomic_uint*)arg0) + (v3 >> 2);\n"
            "  uint v5 = static_cast<uint>((v3 & 3ul) * 8ul);\n"
            "  uint v6 = 255u << v5;\n"
            "  uint v7 = atomic_load_explicit(v4, memory_order_relaxed);\n"
            "  bool v8 = false;\n"
            "  do {\n"
            "    char v9 = static_cast<char>((v7 >> v5) & 255u);\n"
            "    char v10 = static_cast<char>(15);\n"
            "    char v11 = v9 & v10;\n"
            "    char v12 = v11 | v1;\n"
            "    uint v13 = (static_cast<uint>(v12) & 255u) << v5;\n"
            "    uint v14 = (v7 & ~v6) | v13;\n"
            "    v8 = atomic_compare_exchange_weak_explicit(v4, &v7, v14, memory_order_relaxed, memory_order_relaxed);\n"
            "  } while (!v8);\n"
            "}\n");
}

TEST(TranslateToMSL, EmitsIntegerAddAndBoolConstant) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @sum(%a: tensor<1xi32> {xla.slice_index = 0 : i64},
                     %c: tensor<1xi32> {xla.slice_index = 1 : i64})
          -> tensor<1xi32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<1xi32>
        %k = arith.constant 5 : i32
        %y = arith.addi %x, %k : i32
        %out = tensor.insert %y into %c[%i] : tensor<1xi32>
        return %out : tensor<1xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("int v3 = v1 + v2"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("int v2 = static_cast<int>(5)"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsCountLeadingZeros) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @clz(%a: tensor<1xi32> {xla.slice_index = 0 : i64},
                     %c: tensor<1xi32> {xla.slice_index = 1 : i64})
          -> tensor<1xi32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<1xi32>
        %y = math.ctlz %x : i32
        %out = tensor.insert %y into %c[%i] : tensor<1xi32>
        return %out : tensor<1xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find(
                "int v2 = static_cast<int>(metal::clz(static_cast<uint>(v1)))"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, IndexCastUIZeroExtendsSignlessInput) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @index_castui(%a: tensor<1xi8> {xla.slice_index = 0 : i64},
                              %c: tensor<1xi64> {xla.slice_index = 1 : i64})
          -> tensor<1xi64> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<1xi8>
        %idx = arith.index_castui %x : i8 to index
        %y = arith.index_cast %idx : index to i64
        %out = tensor.insert %y into %c[%i] : tensor<1xi64>
        return %out : tensor<1xi64>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find(
                "long v2 = static_cast<long>(static_cast<uchar>(v1));"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsSpecialFloatValues) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @special(%out: tensor<3xf32> {xla.slice_index = 0 : i64})
          -> tensor<3xf32> attributes {xla.entry} {
        %i0 = arith.constant 0 : index
        %i1 = arith.constant 1 : index
        %i2 = arith.constant 2 : index
        %inf = arith.constant 0x7F800000 : f32
        %ninf = arith.constant 0xFF800000 : f32
        %nan = arith.constant 0x7FC00000 : f32
        %t0 = tensor.insert %inf into %out[%i0] : tensor<3xf32>
        %t1 = tensor.insert %ninf into %t0[%i1] : tensor<3xf32>
        %t2 = tensor.insert %nan into %t1[%i2] : tensor<3xf32>
        return %t2 : tensor<3xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // Type-generic via metal::numeric_limits<T>.
  EXPECT_NE(result.source().find("= metal::numeric_limits<float>::infinity();"),
            std::string::npos)
      << result.source();
  EXPECT_NE(
      result.source().find("= (-metal::numeric_limits<float>::infinity());"),
      std::string::npos)
      << result.source();
  EXPECT_NE(
      result.source().find("= metal::numeric_limits<float>::quiet_NaN();"),
      std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsHalfSpecialValues) {
  auto ctx = MakeMlirContext();
  // MSL has the per-type HUGE_VALH macro for half infinity (no per-type
  // NaN macro exists; we cast float NAN). Half bit patterns: 0x7C00 = +inf,
  // 0xFC00 = -inf, 0x7E00 = qNaN.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @half_special(%out: tensor<3xf16> {xla.slice_index = 0 : i64})
          -> tensor<3xf16> attributes {xla.entry} {
        %i0 = arith.constant 0 : index
        %i1 = arith.constant 1 : index
        %i2 = arith.constant 2 : index
        %inf = arith.constant 0x7C00 : f16
        %ninf = arith.constant 0xFC00 : f16
        %nan = arith.constant 0x7E00 : f16
        %t0 = tensor.insert %inf into %out[%i0] : tensor<3xf16>
        %t1 = tensor.insert %ninf into %t0[%i1] : tensor<3xf16>
        %t2 = tensor.insert %nan into %t1[%i2] : tensor<3xf16>
        return %t2 : tensor<3xf16>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("= metal::numeric_limits<half>::infinity();"),
            std::string::npos)
      << result.source();
  EXPECT_NE(
      result.source().find("= (-metal::numeric_limits<half>::infinity());"),
      std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("= metal::numeric_limits<half>::quiet_NaN();"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsStaticCastFor64BitIntegers) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @big_int(%a: tensor<1xi64> {xla.slice_index = 0 : i64},
                         %c: tensor<1xi64> {xla.slice_index = 1 : i64})
          -> tensor<1xi64> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<1xi64>
        %k = arith.constant 5000000000 : i64
        %y = arith.addi %x, %k : i64
        %out = tensor.insert %y into %c[%i] : tensor<1xi64>
        return %out : tensor<1xi64>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // 64-bit constant emitted as static_cast<long>(<digits>) so the value's
  // final type is pinned at the literal site (the bare decimal's parser
  // type doesn't matter — the cast settles it).
  EXPECT_NE(result.source().find("long v2 = static_cast<long>(5000000000);"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, RoundTripsF32WithFullPrecision) {
  auto ctx = MakeMlirContext();
  // Pi to f32 precision: not exactly representable; needs round-trip-safe
  // formatting (>=9 sig digits) to recover the same f32 bit pattern in MSL.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @pi(%out: tensor<1xf32> {xla.slice_index = 0 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %k = arith.constant 3.14159274 : f32
        %r = tensor.insert %k into %out[%i] : tensor<1xf32>
        return %r : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // The literal should preserve enough digits to recover the same f32 bits.
  EXPECT_NE(result.source().find("3.14159274"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsScfForLoop) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @fill(%out: tensor<8xf32> {xla.slice_index = 0 : i64})
          -> tensor<8xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %v = arith.constant 1.0 : f32
        %r = scf.for %i = %c0 to %c8 step %c1
            iter_args(%dst = %out) -> tensor<8xf32> {
          %new = tensor.insert %v into %dst[%i] : tensor<8xf32>
          scf.yield %new : tensor<8xf32>
        }
        return %r : tensor<8xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // Walk through expected lines: counted-loop syntax with the induction
  // variable, body writing to the destination via the iter_arg alias.
  EXPECT_NE(result.source().find("for (long v4 = v0; v4 < v1; v4 += v2) {\n"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("arg0[v4] = v3;"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsScfIfWithoutResults) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @gated(%out: tensor<1xf32> {xla.slice_index = 0 : i64},
                       %flag: tensor<1xi1> {xla.slice_index = 1 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %cond = tensor.extract %flag[%c0] : tensor<1xi1>
        scf.if %cond {
          %v = arith.constant 42.0 : f32
          %t = tensor.insert %v into %out[%c0] : tensor<1xf32>
        }
        return %out : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("if (v1) {"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("arg0[v0] = v2;"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, RejectsScfForWithReassignedIterArg) {
  auto ctx = MakeMlirContext();
  // The iter_arg %acc is reassigned to a fresh scalar (%next) before
  // yielding, so the scf.yield value is not aliased to %acc. The current
  // emitter only supports iter_args whose yield aliases the iter_arg.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @sum(%out: tensor<1xi32> {xla.slice_index = 0 : i64})
          -> tensor<1xi32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %cstep = arith.constant 1 : index
        %init = arith.constant 0 : i32
        %sum = scf.for %i = %c0 to %c4 step %cstep
            iter_args(%acc = %init) -> i32 {
          %one = arith.constant 1 : i32
          %next = arith.addi %acc, %one : i32
          scf.yield %next : i32
        }
        %final = tensor.insert %sum into %out[%c0] : tensor<1xi32>
        return %final : tensor<1xi32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  auto status = TranslateToMSL(*module).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented) << status;
  EXPECT_NE(status.message().find("scf.for"), absl::string_view::npos)
      << status;
}

TEST(TranslateToMSL, AddsKernelAttributeParamsForGpuOps) {
  auto ctx = MakeMlirContext();
  // Uses gpu.thread_id, gpu.block_id, gpu.block_dim — should pull in three
  // [[...]] kernel-attribute parameters and lower each op to a local-var
  // declaration that aliases the appropriate component.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @par(%out: tensor<256xf32> {xla.slice_index = 0 : i64})
          -> tensor<256xf32> attributes {xla.entry} {
        %t = gpu.thread_id x
        %b = gpu.block_id x
        %d = gpu.block_dim x
        %v = arith.constant 1.0 : f32
        %r = tensor.insert %v into %out[%t] : tensor<256xf32>
        return %r : tensor<256xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(
      result.source().find("uint3 tid [[thread_position_in_threadgroup]]"),
      std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("uint3 bid [[threadgroup_position_in_grid]]"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("uint3 block_dim [[threads_per_threadgroup]]"),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("static_cast<long>(tid.x)"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("static_cast<long>(bid.x)"), std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("static_cast<long>(block_dim.x)"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, OmitsUnusedKernelAttributeParams) {
  auto ctx = MakeMlirContext();
  // No gpu.* ops at all — no [[thread_position_*]] etc. should appear.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @nokid(%out: tensor<1xf32> {xla.slice_index = 0 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %v = arith.constant 0.0 : f32
        %r = tensor.insert %v into %out[%i] : tensor<1xf32>
        return %r : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_EQ(result.source().find("[[thread_position_"), std::string::npos)
      << result.source();
  EXPECT_EQ(result.source().find("[[threadgroup_position_"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsGpuBarrier) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @sync(%out: tensor<1xf32> {xla.slice_index = 0 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %v = arith.constant 1.0 : f32
        %r = tensor.insert %v into %out[%i] : tensor<1xf32>
        gpu.barrier
        return %r : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, DetectsGpuOpsInsideScfFor) {
  auto ctx = MakeMlirContext();
  // gpu.thread_id inside a loop body should still be picked up by the
  // pre-pass; the attribute parameter must appear in the signature.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @inloop(%out: tensor<4xf32> {xla.slice_index = 0 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %v = arith.constant 1.0 : f32
        %r = scf.for %i = %c0 to %c4 step %c1
            iter_args(%dst = %out) -> tensor<4xf32> {
          %t = gpu.thread_id x
          %new = tensor.insert %v into %dst[%t] : tensor<4xf32>
          scf.yield %new : tensor<4xf32>
        }
        return %r : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(
      result.source().find("uint3 tid [[thread_position_in_threadgroup]]"),
      std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, HoistsAllocateSharedToFunctionScope) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @sh(%out: tensor<64xf32> {xla.slice_index = 0 : i64})
          -> tensor<64xf32> attributes {xla.entry} {
        %shared = xla_gpu.allocate_shared : tensor<64xf32>
        %t = gpu.thread_id x
        %v = arith.constant 1.0 : f32
        %w = tensor.insert %v into %shared[%t] : tensor<64xf32>
        return %out : tensor<64xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // Declaration appears at the top of the kernel body.
  EXPECT_NE(result.source().find("threadgroup float v0[64];"),
            std::string::npos)
      << result.source();
  // And the body writes into v0 (the allocate_shared result).
  EXPECT_NE(result.source().find("v0[v1] = v2;"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsSyncThreadsAsBarrierAndAliasesOperands) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @synced(%out: tensor<8xf32> {xla.slice_index = 0 : i64})
          -> tensor<8xf32> attributes {xla.entry} {
        %shared = xla_gpu.allocate_shared : tensor<8xf32>
        %t = gpu.thread_id x
        %v = arith.constant 1.0 : f32
        %after_write = tensor.insert %v into %shared[%t] : tensor<8xf32>
        %synced = xla_gpu.sync_threads %after_write : tensor<8xf32>
        %r = tensor.extract %synced[%t] : tensor<8xf32>
        %final = tensor.insert %r into %out[%t] : tensor<8xf32>
        return %final : tensor<8xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"),
            std::string::npos)
      << result.source();
  // After sync_threads, reads on its result must hit the same shared array
  // (v0) — not a freshly-named variable.
  EXPECT_NE(result.source().find("float v3 = v0[v1];"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsVectorTransferReadAndExtract) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @vread(%a: tensor<16xf32> {xla.slice_index = 0 : i64},
                       %b: tensor<1xf32> {xla.slice_index = 1 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %pad = arith.constant 0.0 : f32
        %v = vector.transfer_read %a[%c0], %pad : tensor<16xf32>, vector<4xf32>
        %s = vector.extract %v[2] : f32 from vector<4xf32>
        %r = tensor.insert %s into %b[%c0] : tensor<1xf32>
        return %r : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  // Vector load goes through a const device float4* reinterpret-cast.
  EXPECT_NE(
      result.source().find("float4 v2 = *(const device float4*)(&arg0[v0]);"),
      std::string::npos)
      << result.source();
  // Static-position extract becomes a subscript.
  EXPECT_NE(result.source().find("float v3 = v2[2];"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsVectorTransferWriteRoundTrip) {
  auto ctx = MakeMlirContext();
  // Copy a float4 chunk from one device buffer to another.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @vcopy(%src: tensor<32xf32> {xla.slice_index = 0 : i64},
                       %dst: tensor<32xf32> {xla.slice_index = 1 : i64})
          -> tensor<32xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %pad = arith.constant 0.0 : f32
        %v = vector.transfer_read %src[%c0], %pad
            : tensor<32xf32>, vector<4xf32>
        %r = vector.transfer_write %v, %dst[%c0]
            : vector<4xf32>, tensor<32xf32>
        return %r : tensor<32xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(
      result.source().find("float4 v2 = *(const device float4*)(&arg0[v0]);"),
      std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("*(device float4*)(&arg1[v0]) = v2;"),
            std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, EmitsVectorOnThreadgroupAddressSpace) {
  auto ctx = MakeMlirContext();
  // Source is an allocate_shared buffer — the reinterpret cast must use
  // the `threadgroup` address space, not `device`.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @vshared(%dst: tensor<4xf32> {xla.slice_index = 0 : i64})
          -> tensor<4xf32> attributes {xla.entry} {
        %shared = xla_gpu.allocate_shared : tensor<32xf32>
        %c0 = arith.constant 0 : index
        %pad = arith.constant 0.0 : f32
        %v = vector.transfer_read %shared[%c0], %pad
            : tensor<32xf32>, vector<4xf32>
        %r = vector.transfer_write %v, %dst[%c0]
            : vector<4xf32>, tensor<4xf32>
        return %r : tensor<4xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("(const threadgroup float4*)(&v0["),
            std::string::npos)
      << result.source();
  EXPECT_NE(result.source().find("(device float4*)(&arg0["), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, RejectsUnsupportedVectorWidth) {
  auto ctx = MakeMlirContext();
  // vector<5xf32> — MSL only supports widths 2, 3, 4.
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @vbad(%a: tensor<16xf32> {xla.slice_index = 0 : i64},
                      %b: tensor<5xf32> {xla.slice_index = 1 : i64})
          -> tensor<5xf32> attributes {xla.entry} {
        %c0 = arith.constant 0 : index
        %pad = arith.constant 0.0 : f32
        %v = vector.transfer_read %a[%c0], %pad
            : tensor<16xf32>, vector<5xf32>
        %r = vector.transfer_write %v, %b[%c0]
            : vector<5xf32>, tensor<5xf32>
        return %r : tensor<5xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  auto status = TranslateToMSL(*module).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented) << status;
  EXPECT_NE(status.message().find("vector width"), absl::string_view::npos)
      << status;
}

TEST(TranslateToMSL, EmitsFloatingRemainder) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @mod(%a: tensor<1xf32> {xla.slice_index = 0 : i64},
                     %c: tensor<1xf32> {xla.slice_index = 1 : i64})
          -> tensor<1xf32> attributes {xla.entry} {
        %i = arith.constant 0 : index
        %x = tensor.extract %a[%i] : tensor<1xf32>
        %k = arith.constant 3.000000e+00 : f32
        %y = arith.remf %x, %k : f32
        %out = tensor.insert %y into %c[%i] : tensor<1xf32>
        return %out : tensor<1xf32>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(MslKernelSource result, TranslateToMSL(*module));
  EXPECT_NE(result.source().find("metal::precise::fmod"), std::string::npos)
      << result.source();
}

TEST(TranslateToMSL, RejectsF64BecauseAppleSiliconHasNoFp64) {
  auto ctx = MakeMlirContext();
  constexpr absl::string_view kInput = R"mlir(
    module {
      func.func @f64_kernel(%a: tensor<4xf64> {xla.slice_index = 0 : i64})
          -> tensor<4xf64> attributes {xla.entry} {
        return %a : tensor<4xf64>
      }
    }
  )mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kInput, ctx.get());
  ASSERT_TRUE(module);

  auto status = TranslateToMSL(*module).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("f64"), absl::string_view::npos);
}

}  // namespace
}  // namespace metal
}  // namespace xla
