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

#ifndef XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES
#define XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerFloatStoragePass
    : Pass<"xla-metal-lower-float-storage", "mlir::ModuleOp"> {
  let summary = "Lowers unsupported low-precision float storage for Metal.";

  let description = [{
    Converts unsupported low-precision scalar, vector, and tensor float types
    to integer storage bit patterns. BF16 lowers to i16 and 8-bit floats lower
    to i8. The pass also expands BF16<->F32 casts into integer bit operations.
    This is a Metal codegen compatibility pass for MSL, which does not expose
    these types as native scalar types.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::ub::UBDialect",
    "mlir::vector::VectorDialect",
  ];

  let constructor = "CreateLowerFloatStoragePass()";
}

def LowerSubByteStoragePass
    : Pass<"xla-metal-lower-sub-byte-storage", "mlir::ModuleOp"> {
  let summary = "Lowers sub-byte storage for Metal.";

  let description = [{
    Rewrites i2/i4/f4 tensors to packed i8 tensors and rewrites scalar/vector
    sub-byte values to i8 low-bit storage. This is a Metal-local fork of the
    sub-byte machinery in lower_tensors, targeted to the tensor/arith/vector
    dialect subset consumed by the MSL translator instead of LLVM pointer IR.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::ub::UBDialect",
    "mlir::vector::VectorDialect",
    "::xla::XlaDialect",
  ];

  let constructor = "CreateLowerSubByteStoragePass()";
}

def ExpandFloatOpsPass
    : Pass<"xla-metal-expand-float-ops", "mlir::ModuleOp"> {
  let summary = "Expands float operations unsupported by Metal.";

  let description = [{
    Rewrites floating-point operations that have no direct MSL spelling into
    arithmetic and math operations that the MSL translator already supports.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect",
  ];

  let constructor = "CreateExpandFloatOpsPass()";
}

def ConvertComplexToArithMathPass
    : Pass<"xla-metal-convert-complex-to-arith-math", "mlir::ModuleOp"> {
  let summary = "Lowers Complex dialect operations for Metal.";

  let description = [{
    Converts scalar complex values to vector<2xT> values and lowers Complex
    dialect operations to arith, math, tensor, and vector operations understood
    by the Metal MSL path. Complex tensors are lowered to flat real/imag
    element storage to match XLA's interleaved complex buffer layout.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::complex::ComplexDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::ub::UBDialect",
    "mlir::vector::VectorDialect",
  ];

  let constructor = "CreateConvertComplexToArithMathPass()";
}

#endif  // XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES
