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

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/IndentedOstream.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/name_uniquer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace metal {
namespace {

constexpr absl::string_view kEntryAttrName = "xla.entry";
constexpr absl::string_view kSliceIndexAttrName = "xla.slice_index";

std::string GetUniqueMslFunctionName(absl::string_view name,
                                     NameUniquer* function_name_uniquer) {
  CHECK(function_name_uniquer != nullptr);
  return function_name_uniquer->GetUniqueName(
      llvm_ir::SanitizeFunctionName(std::string(name)));
}

std::string GetUniqueMslHelperName(llvm::StringRef entry_name,
                                   llvm::StringRef helper_name,
                                   NameUniquer* function_name_uniquer) {
  return GetUniqueMslFunctionName(
      absl::StrCat(entry_name.str(), "_", helper_name.str()),
      function_name_uniquer);
}

bool IsSingleElementVector(mlir::Type type) {
  auto vector_type = mlir::dyn_cast<mlir::VectorType>(type);
  return vector_type && vector_type.getRank() == 1 &&
         vector_type.getDimSize(0) == 1;
}

std::optional<int64_t> GetStaticVectorIndex(
    mlir::ArrayRef<mlir::OpFoldResult> position) {
  if (position.size() != 1) return std::nullopt;
  auto attr = mlir::dyn_cast<mlir::Attribute>(position[0]);
  if (!attr) return std::nullopt;
  auto integer_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
  if (!integer_attr) return std::nullopt;
  return integer_attr.getInt();
}

// Maps an MLIR element type to its MSL spelling.
absl::StatusOr<std::string> EmitElementType(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    const unsigned width = integer.getWidth();
    const bool is_unsigned = integer.isUnsigned();
    switch (width) {
      case 1:
        return std::string("bool");
      case 8:
        return std::string(is_unsigned ? "uchar" : "char");
      case 16:
        return std::string(is_unsigned ? "ushort" : "short");
      case 32:
        return std::string(is_unsigned ? "uint" : "int");
      case 64:
        return std::string(is_unsigned ? "ulong" : "long");
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported integer width for MSL: ", width));
  }
  if (mlir::isa<mlir::Float16Type>(type)) {
    return std::string("half");
  }
  if (mlir::isa<mlir::Float32Type>(type)) {
    return std::string("float");
  }
  if (mlir::isa<mlir::Float64Type>(type)) {
    return absl::InvalidArgumentError(
        "Apple Silicon GPUs have no fp64 hardware; f64 is not supported.");
  }
  if (mlir::isa<mlir::BFloat16Type>(type)) {
    return absl::InvalidArgumentError("bf16 requires Metal 3 (macOS 13+).");
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported element type for MSL: ", mlir::debugString(type)));
}

absl::StatusOr<std::string> TypeToMSL(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type)) {
    return std::string("long");
  }
  if (auto vec = mlir::dyn_cast<mlir::VectorType>(type)) {
    if (vec.getRank() != 1) {
      return absl::UnimplementedError(
          "Multi-rank vector types are not yet supported by the MSL emitter.");
    }
    const int64_t n = vec.getDimSize(0);
    if (n == 1) {
      // Metal has no float1/int1 vector spelling. Treat MLIR's single-lane
      // vectors as scalars; vector ops that need lane semantics special-case
      // vector<1xT> in their emitters.
      return EmitElementType(vec.getElementType());
    }
    if (n != 2 && n != 3 && n != 4) {
      return absl::UnimplementedError(
          absl::StrCat("MSL vector width must be 2, 3, or 4; got ", n));
    }
    TF_ASSIGN_OR_RETURN(std::string elem,
                        EmitElementType(vec.getElementType()));
    return absl::StrCat(elem, n);
  }
  return EmitElementType(type);
}

// The MSL spelling of `type` reinterpreted as unsigned. MLIR integers are
// signless and map to MSL's signed spellings, so operations with unsigned
// semantics (u* comparisons, minui, extui, ...) cast through this first.
absl::StatusOr<std::string> UnsignedMslType(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type)) {
    return std::string("ulong");
  }
  auto int_ty = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!int_ty) {
    return absl::UnimplementedError(
        absl::StrCat("UnsignedMslType: expected integer/index type, got ",
                     mlir::debugString(type)));
  }
  return EmitElementType(mlir::IntegerType::get(
      type.getContext(), int_ty.getWidth(), mlir::IntegerType::Unsigned));
}

absl::StatusOr<int64_t> SliceIndexOf(mlir::func::FuncOp func, unsigned idx) {
  auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(
      idx, std::string(kSliceIndexAttrName));
  if (!attr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Argument #", idx,
        " is missing the xla.slice_index attribute needed to derive the MSL "
        "buffer-binding index."));
  }
  return attr.getInt();
}

// Formats an `arith.constant`'s value as an MSL literal of the given type.
absl::StatusOr<std::string> FormatConstantLiteral(mlir::Type type,
                                                  mlir::Attribute value) {
  if (mlir::isa<mlir::IndexType>(type)) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(value);
    if (!attr) {
      return absl::InvalidArgumentError(
          "arith.constant of index type expected an IntegerAttr value.");
    }
    return absl::StrCat("static_cast<long>(", attr.getInt(), ")");
  }
  if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(value);
    if (!attr) {
      return absl::InvalidArgumentError(
          "arith.constant of integer type expected an IntegerAttr value.");
    }
    if (int_ty.getWidth() == 1) {
      return std::string(attr.getInt() ? "true" : "false");
    }
    // static_cast pins the literal's type at the site, avoiding narrowing
    // diagnostics on edge values like INT_MIN.
    TF_ASSIGN_OR_RETURN(std::string elem_msl, EmitElementType(type));
    std::string digits = int_ty.isUnsigned() ? std::to_string(attr.getUInt())
                                             : std::to_string(attr.getInt());
    return absl::StrCat("static_cast<", elem_msl, ">(", digits, ")");
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto attr = mlir::dyn_cast<mlir::FloatAttr>(value);
    if (!attr) {
      return absl::InvalidArgumentError(
          "arith.constant of float type expected a FloatAttr value.");
    }
    absl::string_view suffix;
    int precision;
    if (mlir::isa<mlir::Float16Type>(type)) {
      suffix = "h";
      precision = 5;  // round-trips IEEE half.
    } else if (mlir::isa<mlir::Float32Type>(type)) {
      suffix = "f";
      precision = 9;  // round-trips IEEE single.
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("arith.constant of unsupported float type: ",
                       mlir::debugString(type)));
    }

    // Widen to double for absl::StrFormat. Lossless for f16 and f32.
    llvm::APFloat val = attr.getValue();
    bool loses_info = false;
    val.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    const double d = val.convertToDouble();

    if (std::isnan(d) || std::isinf(d)) {
      TF_ASSIGN_OR_RETURN(std::string elem_msl, EmitElementType(type));
      absl::string_view method = std::isnan(d) ? "quiet_NaN()" : "infinity()";
      std::string base =
          absl::StrCat("metal::numeric_limits<", elem_msl, ">::", method);
      if (std::isinf(d) && std::signbit(d)) {
        return absl::StrCat("(-", base, ")");
      }
      return base;
    }

    // `#` forces a decimal point so integer-valued floats render as
    // `1.00000000f`, not the bare `1f` that wouldn't parse.
    return absl::StrCat(absl::StrFormat("%#.*g", precision, d), suffix);
  }
  if (auto vec_ty = mlir::dyn_cast<mlir::VectorType>(type)) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(value);
    if (!dense) {
      return absl::InvalidArgumentError(absl::StrCat(
          "arith.constant of vector type expected a DenseElementsAttr value: ",
          mlir::debugString(type)));
    }
    TF_ASSIGN_OR_RETURN(std::string vec_msl, TypeToMSL(type));
    if (dense.isSplat()) {
      TF_ASSIGN_OR_RETURN(
          std::string lit,
          FormatConstantLiteral(vec_ty.getElementType(),
                                dense.getSplatValue<mlir::Attribute>()));
      return absl::StrCat(vec_msl, "(", lit, ")");
    }
    std::string out = absl::StrCat(vec_msl, "(");
    bool first = true;
    for (mlir::Attribute elt : dense.getValues<mlir::Attribute>()) {
      if (!first) out.append(", ");
      first = false;
      TF_ASSIGN_OR_RETURN(std::string lit,
                          FormatConstantLiteral(vec_ty.getElementType(), elt));
      out.append(lit);
    }
    out.append(")");
    return out;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "arith.constant of unsupported type: ", mlir::debugString(type)));
}

// Emits an MSL function body statement-by-statement, threading a name table
// from MLIR SSA values to MSL variable identifiers.
class MslEmitter {
 public:
  explicit MslEmitter(
      mlir::raw_indented_ostream& os,
      llvm::StringMap<std::string> emitted_function_names = {},
      llvm::StringMap<std::string> emitted_result_type_names = {})
      : os_(os),
        emitted_function_names_(std::move(emitted_function_names)),
        emitted_result_type_names_(std::move(emitted_result_type_names)) {}

  // The entry: a `kernel void` whose results land in output buffers.
  absl::Status EmitFunction(mlir::func::FuncOp func) {
    in_entry_ = true;
    GpuAttrs attrs = DetectGpuAttrs(func);
    TF_RETURN_IF_ERROR(EmitSignature(func, attrs));
    os_.indent();
    TF_RETURN_IF_ERROR(HoistSharedAllocations(func));
    for (mlir::Operation& op : func.getBody().front()) {
      TF_RETURN_IF_ERROR(EmitOp(&op));
    }
    os_.unindent();
    os_ << "}\n";
    return absl::OkStatus();
  }

  // A non-entry subcomputation, emitted as a plain MSL device function whose
  // func.return yields a value. Reached via func.call from the kernel (or
  // another device function) after ConvertPureCallOps + the inliner.
  absl::Status EmitDeviceFunction(mlir::func::FuncOp func) {
    in_entry_ = false;
    TF_RETURN_IF_ERROR(EmitDeviceSignature(func, /*names=*/true));
    os_ << " {\n";
    os_.indent();
    for (mlir::Operation& op : func.getBody().front()) {
      TF_RETURN_IF_ERROR(EmitOp(&op));
    }
    os_.unindent();
    os_ << "}\n";
    return absl::OkStatus();
  }

  // Forward declaration, so device functions may call each other regardless of
  // emission order.
  absl::Status EmitDeviceFunctionDecl(mlir::func::FuncOp func) {
    TF_RETURN_IF_ERROR(EmitDeviceSignature(func, /*names=*/false));
    os_ << ";\n";
    return absl::OkStatus();
  }

  absl::Status EmitDeviceFunctionResultType(mlir::func::FuncOp func) {
    auto results = func.getFunctionType().getResults();
    if (results.size() < 2) return absl::OkStatus();
    os_ << "struct " << EmittedFunctionResultTypeName(func) << " {\n";
    os_.indent();
    for (auto [i, result] : llvm::enumerate(results)) {
      TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(result));
      os_ << ty << " result" << i << ";\n";
    }
    os_.unindent();
    os_ << "};\n";
    return absl::OkStatus();
  }

  absl::Status EmitFuncCall(mlir::func::CallOp op) {
    std::string call = absl::StrCat(EmittedFunctionName(op.getCallee()), "(");
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0) call += ", ";
      TF_ASSIGN_OR_RETURN(std::string a, GetName(op.getOperand(i)));
      call += a;
    }
    call += ")";
    if (op.getNumResults() == 0) {
      os_ << call << ";\n";
      return absl::OkStatus();
    }
    if (op.getNumResults() == 1) {
      TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op.getResult(0).getType()));
      std::string name = BindValueName(op.getResult(0));
      os_ << ty << " " << name << " = " << call << ";\n";
      return absl::OkStatus();
    }
    std::string result_type = EmittedFunctionResultTypeName(op.getCallee());
    std::string result = CreateFreshName();
    os_ << result_type << " " << result << " = " << call << ";\n";
    for (auto [i, value] : llvm::enumerate(op.getResults())) {
      TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(value.getType()));
      std::string name = BindValueName(value);
      os_ << ty << " " << name << " = " << result << ".result" << i << ";\n";
    }
    return absl::OkStatus();
  }

  absl::Status EmitFuncReturn(mlir::func::ReturnOp op) {
    // In the entry kernel, results are written to output buffers in place, so
    // the returned tensor values have no MSL counterpart.
    if (in_entry_) return absl::OkStatus();
    if (op.getNumOperands() == 0) {
      os_ << "return;\n";
      return absl::OkStatus();
    }
    if (op.getNumOperands() == 1) {
      TF_ASSIGN_OR_RETURN(std::string v, GetName(op.getOperand(0)));
      os_ << "return " << v << ";\n";
      return absl::OkStatus();
    }
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    CHECK(func != nullptr) << "func.return is not nested in func.func";
    os_ << "return " << EmittedFunctionResultTypeName(func) << "{";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0) os_ << ", ";
      TF_ASSIGN_OR_RETURN(std::string v, GetName(op.getOperand(i)));
      os_ << v;
    }
    os_ << "};\n";
    return absl::OkStatus();
  }

 private:
  // MSL kernel-attribute parameters that the body references. Only those
  // actually used appear in the signature.
  struct GpuAttrs {
    bool thread_id = false;
    bool block_id = false;
    bool block_dim = false;
    bool grid_dim = false;
  };

  static GpuAttrs DetectGpuAttrs(mlir::func::FuncOp func) {
    GpuAttrs attrs;
    func.walk([&](mlir::Operation* op) {
      if (mlir::isa<mlir::gpu::ThreadIdOp>(op)) attrs.thread_id = true;
      if (mlir::isa<mlir::gpu::BlockIdOp>(op)) attrs.block_id = true;
      if (mlir::isa<mlir::gpu::BlockDimOp>(op)) attrs.block_dim = true;
      if (mlir::isa<mlir::gpu::GridDimOp>(op)) attrs.grid_dim = true;
    });
    return attrs;
  }

  absl::Status EmitSignature(mlir::func::FuncOp func, const GpuAttrs& attrs) {
    os_ << "kernel void " << EmittedFunctionName(func) << "(";
    bool first = true;
    auto emit_param = [&](absl::string_view decl) {
      if (!first) os_ << ",";
      first = false;
      os_ << "\n    " << decl;
    };
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      mlir::Value arg = func.getArgument(i);
      auto tensor_ty = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
      if (!tensor_ty) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Kernel argument #", i, " is not a RankedTensorType; got ",
            mlir::debugString(arg.getType())));
      }
      TF_ASSIGN_OR_RETURN(std::string elem_msl,
                          EmitElementType(tensor_ty.getElementType()));
      // Position-based slot binding: index by arg position, not slice_index.
      // MetalStream::LaunchKernel binds positionally; mismatch would corrupt
      // the encoder when slice_index != position.
      std::string arg_name = absl::StrCat("arg", i);
      BindLValue(arg, arg_name, "device");
      emit_param(absl::StrCat("device ", elem_msl, "* ", arg_name, " [[buffer(",
                              i, ")]]"));
    }
    if (attrs.thread_id) {
      emit_param("uint3 tid [[thread_position_in_threadgroup]]");
    }
    if (attrs.block_id) {
      emit_param("uint3 bid [[threadgroup_position_in_grid]]");
    }
    if (attrs.block_dim) {
      emit_param("uint3 block_dim [[threads_per_threadgroup]]");
    }
    if (attrs.grid_dim) {
      emit_param("uint3 grid_dim [[threadgroups_per_grid]]");
    }
    os_ << ") {\n";
    return absl::OkStatus();
  }

  // Signature for a non-entry device function: `RetType name(params)`, with no
  // trailing brace/semicolon. With `names` the params are named and registered
  // (a definition); without, only types are emitted (a forward declaration).
  // Tensor params map to `device T*` (callers pass device buffers).
  absl::Status EmitDeviceSignature(mlir::func::FuncOp func, bool names) {
    auto results = func.getFunctionType().getResults();
    std::string ret;
    if (results.empty()) {
      ret = "void";
    } else if (results.size() == 1) {
      TF_ASSIGN_OR_RETURN(ret, TypeToMSL(results[0]));
    } else {
      ret = EmittedFunctionResultTypeName(func);
    }
    os_ << ret << " " << EmittedFunctionName(func) << "(";
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (i > 0) os_ << ", ";
      mlir::Value arg = func.getArgument(i);
      std::string pty;
      bool is_buffer = false;
      if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType())) {
        TF_ASSIGN_OR_RETURN(std::string elem,
                            EmitElementType(t.getElementType()));
        pty = absl::StrCat("device ", elem, "*");
        is_buffer = true;
      } else {
        TF_ASSIGN_OR_RETURN(pty, TypeToMSL(arg.getType()));
      }
      os_ << pty;
      if (names) {
        std::string nm = absl::StrCat("arg", i);
        if (is_buffer) {
          BindLValue(arg, nm, "device");
        } else {
          values_[arg] = RValue{nm};
        }
        os_ << " " << nm;
      }
    }
    os_ << ")";
    return absl::OkStatus();
  }

  absl::Status EmitOp(mlir::Operation* op) {
    if (auto co = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      return EmitArithConstant(co);
    }
    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(op)) {
      return EmitBinary(op, "+");
    }
    if (mlir::isa<mlir::arith::SubFOp, mlir::arith::SubIOp>(op)) {
      return EmitBinary(op, "-");
    }
    if (mlir::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(op)) {
      return EmitBinary(op, "*");
    }
    if (mlir::isa<mlir::arith::DivFOp, mlir::arith::DivSIOp>(op)) {
      return EmitBinary(op, "/");
    }
    if (mlir::isa<mlir::arith::DivUIOp>(op)) {
      return EmitArithUnsignedBinary(op, "/");
    }
    if (mlir::isa<mlir::arith::RemSIOp>(op)) {
      return EmitBinary(op, "%");
    }
    if (mlir::isa<mlir::arith::RemUIOp>(op)) {
      return EmitArithUnsignedBinary(op, "%");
    }
    if (mlir::isa<mlir::arith::RemFOp>(op)) {
      return EmitMathCall(op, "metal::precise::fmod");
    }
    if (mlir::isa<mlir::arith::AndIOp>(op)) {
      return EmitBinary(op, "&");
    }
    if (mlir::isa<mlir::arith::OrIOp>(op)) {
      return EmitBinary(op, "|");
    }
    if (mlir::isa<mlir::arith::XOrIOp>(op)) {
      return EmitBinary(op, "^");
    }
    if (mlir::isa<mlir::arith::ShRUIOp>(op)) {
      return EmitArithShrUI(op);
    }
    if (mlir::isa<mlir::arith::ShRSIOp>(op)) {
      // EmitElementType maps signless MLIR ints to signed MSL types, so bare
      // `>>` is already arithmetic (sign-preserving) right shift.
      return EmitBinary(op, ">>");
    }
    if (mlir::isa<mlir::arith::ShLIOp>(op)) {
      return EmitBinary(op, "<<");
    }
    if (mlir::isa<mlir::arith::MinimumFOp>(op)) {
      return EmitArithMinMaxF(op, /*is_min=*/true);
    }
    if (mlir::isa<mlir::arith::MaximumFOp>(op)) {
      return EmitArithMinMaxF(op, /*is_min=*/false);
    }
    if (mlir::isa<mlir::arith::MinUIOp>(op)) {
      return EmitArithUnsignedMinMax(op, "metal::min");
    }
    if (mlir::isa<mlir::arith::MaxUIOp>(op)) {
      return EmitArithUnsignedMinMax(op, "metal::max");
    }
    if (mlir::isa<mlir::arith::MinSIOp>(op)) {
      return EmitMathCall(op, "metal::min");
    }
    if (mlir::isa<mlir::arith::MaxSIOp>(op)) {
      return EmitMathCall(op, "metal::max");
    }
    if (mlir::isa<mlir::arith::NegFOp>(op)) {
      return EmitUnaryOp(op, "-");
    }
    if (auto eu = mlir::dyn_cast<mlir::arith::ExtUIOp>(op)) {
      return EmitArithExtUI(eu);
    }
    if (auto ci = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      return EmitArithCmpI(ci);
    }
    if (auto cf = mlir::dyn_cast<mlir::arith::CmpFOp>(op)) {
      return EmitArithCmpF(cf);
    }
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      return EmitArithSelect(sel);
    }
    if (mlir::isa<mlir::math::AbsFOp>(op)) {
      return EmitMathCall(op, "metal::abs");
    }
    if (mlir::isa<mlir::math::ExpOp>(op)) {
      return EmitMathCall(op, "metal::exp");
    }
    if (mlir::isa<mlir::math::FmaOp>(op)) {
      return EmitMathCall(op, "metal::fma");
    }
    if (mlir::isa<mlir::math::CeilOp>(op)) {
      return EmitMathCall(op, "metal::ceil");
    }
    if (mlir::isa<mlir::math::FloorOp>(op)) {
      return EmitMathCall(op, "metal::floor");
    }
    if (mlir::isa<mlir::math::LogOp>(op)) {
      return EmitMathCall(op, "metal::log");
    }
    if (mlir::isa<mlir::math::SqrtOp>(op)) {
      return EmitMathCall(op, "metal::sqrt");
    }
    if (mlir::isa<mlir::math::PowFOp>(op)) {
      return EmitMathCall(op, "metal::pow");
    }
    if (mlir::isa<mlir::math::CbrtOp>(op)) {
      return EmitMathCbrt(op);
    }
    if (mlir::isa<mlir::math::RsqrtOp>(op)) {
      return EmitMathCall(op, "metal::rsqrt");
    }
    if (mlir::isa<mlir::math::CopySignOp>(op)) {
      return EmitMathCall(op, "metal::copysign");
    }
    if (mlir::isa<mlir::math::Atan2Op>(op)) {
      return EmitMathCall(op, "metal::atan2");
    }
    if (mlir::isa<mlir::math::CosOp>(op)) {
      return EmitMathCall(op, "metal::cos");
    }
    if (mlir::isa<mlir::math::SinOp>(op)) {
      return EmitMathCall(op, "metal::sin");
    }
    if (mlir::isa<mlir::math::TanOp>(op)) {
      return EmitMathCall(op, "metal::tan");
    }
    if (mlir::isa<mlir::math::TanhOp>(op)) {
      return EmitMathCall(op, "metal::tanh");
    }
    if (mlir::isa<mlir::math::RoundOp>(op)) {
      return EmitMathCall(op, "metal::round");
    }
    if (mlir::isa<mlir::math::RoundEvenOp>(op)) {
      // metal::rint rounds halfway cases to even, matching roundTiesToEven.
      return EmitMathCall(op, "metal::rint");
    }
    if (auto clz = mlir::dyn_cast<mlir::math::CountLeadingZerosOp>(op)) {
      return EmitMathCtlz(clz);
    }
    if (auto ctpop = mlir::dyn_cast<mlir::math::CtPopOp>(op)) {
      return EmitMathCtpop(ctpop);
    }
    if (auto ic = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      return EmitArithCast(ic.getOperation());
    }
    if (auto ic = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(op)) {
      return EmitArithIndexCastUI(ic);
    }
    if (auto sf = mlir::dyn_cast<mlir::arith::SIToFPOp>(op)) {
      return EmitArithCast(sf.getOperation());
    }
    // Plain static_cast conversions: float widen/narrow, signed-int
    // widen/narrow, and float->signed-int (truncates toward zero).
    if (auto c = mlir::dyn_cast<mlir::arith::ExtFOp>(op)) {
      return EmitArithCast(c.getOperation());
    }
    if (auto c = mlir::dyn_cast<mlir::arith::TruncFOp>(op)) {
      return EmitArithCast(c.getOperation());
    }
    if (auto c = mlir::dyn_cast<mlir::arith::ExtSIOp>(op)) {
      return EmitArithCast(c.getOperation());
    }
    if (auto c = mlir::dyn_cast<mlir::arith::TruncIOp>(op)) {
      return EmitArithCast(c.getOperation());
    }
    if (auto c = mlir::dyn_cast<mlir::arith::FPToSIOp>(op)) {
      return EmitArithCast(c.getOperation());
    }
    if (auto c = mlir::dyn_cast<mlir::arith::FPToUIOp>(op)) {
      return EmitArithFPToUI(c);
    }
    if (auto c = mlir::dyn_cast<mlir::arith::UIToFPOp>(op)) {
      return EmitArithUIToFP(c);
    }
    if (auto c = mlir::dyn_cast<mlir::arith::BitcastOp>(op)) {
      return EmitArithBitcast(c);
    }
    if (auto ex = mlir::dyn_cast<mlir::tensor::ExtractOp>(op)) {
      return EmitTensorExtract(ex);
    }
    if (auto in = mlir::dyn_cast<mlir::tensor::InsertOp>(op)) {
      return EmitTensorInsert(in);
    }
    if (auto atomic = mlir::dyn_cast<::xla::AtomicRMWOp>(op)) {
      return EmitXlaAtomicRMW(atomic);
    }
    if (auto fo = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      return EmitScfFor(fo);
    }
    if (auto ifo = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      return EmitScfIf(ifo);
    }
    if (auto iso = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op)) {
      return EmitScfIndexSwitch(iso);
    }
    if (auto t = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
      return EmitDimComponent(t.getResult(), "tid", t.getDimension());
    }
    if (auto b = mlir::dyn_cast<mlir::gpu::BlockIdOp>(op)) {
      return EmitDimComponent(b.getResult(), "bid", b.getDimension());
    }
    if (auto bd = mlir::dyn_cast<mlir::gpu::BlockDimOp>(op)) {
      return EmitDimComponent(bd.getResult(), "block_dim", bd.getDimension());
    }
    if (auto gd = mlir::dyn_cast<mlir::gpu::GridDimOp>(op)) {
      return EmitDimComponent(gd.getResult(), "grid_dim", gd.getDimension());
    }
    if (mlir::isa<mlir::gpu::BarrierOp>(op)) {
      os_ << "threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n";
      return absl::OkStatus();
    }
    if (auto sh = mlir::dyn_cast<mlir::gpu::ShuffleOp>(op)) {
      return EmitGpuShuffle(sh);
    }
    if (mlir::isa<::xla::gpu::AllocateSharedOp>(op)) {
      // Hoisted to function scope by HoistSharedAllocations.
      return absl::OkStatus();
    }
    if (auto sync = mlir::dyn_cast<::xla::gpu::SyncThreadsOp>(op)) {
      return EmitSyncThreads(sync);
    }
    if (auto vr = mlir::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      return EmitVectorTransferRead(vr);
    }
    if (auto vw = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      return EmitVectorTransferWrite(vw);
    }
    if (auto vb = mlir::dyn_cast<mlir::vector::BitCastOp>(op)) {
      return EmitVectorBitcast(vb);
    }
    if (auto ve = mlir::dyn_cast<mlir::vector::ExtractOp>(op)) {
      return EmitVectorExtract(ve);
    }
    if (auto vi = mlir::dyn_cast<mlir::vector::InsertOp>(op)) {
      return EmitVectorInsert(vi);
    }
    if (auto vf = mlir::dyn_cast<mlir::vector::FromElementsOp>(op)) {
      return EmitVectorFromElements(vf);
    }
    if (auto vb = mlir::dyn_cast<mlir::vector::BroadcastOp>(op)) {
      return EmitVectorBroadcast(vb);
    }
    if (auto po = mlir::dyn_cast<mlir::ub::PoisonOp>(op)) {
      return EmitUbPoison(po);
    }
    if (mlir::isa<mlir::scf::YieldOp>(op)) {
      // Yielded values are wired in EmitScfFor/EmitScfIf via the name table.
      return absl::OkStatus();
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      return EmitFuncCall(call);
    }
    if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
      return EmitFuncReturn(ret);
    }
    return absl::UnimplementedError(
        absl::StrCat("MSL emitter does not yet handle op: ",
                     op->getName().getStringRef().str()));
  }

  absl::Status EmitArithConstant(mlir::arith::ConstantOp op) {
    mlir::Type ty = op.getType();
    // Materialise non-scalar tensor constants as local C-style arrays. The
    // LLVM GPU pipeline does this in LowerTensorsPass (RewriteNonScalarConstants),
    // which we don't run; downstream tensor.extract becomes plain `arr[i]`.
    if (auto tensor_ty = mlir::dyn_cast<mlir::RankedTensorType>(ty)) {
      if (tensor_ty.getRank() > 1) {
        return absl::UnimplementedError(absl::StrCat(
            "arith.constant of tensor type must be rank-0 or rank-1 after "
            "flatten; got ",
            mlir::debugString(ty)));
      }
      auto dense_attr =
          mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValueAttr());
      if (!dense_attr) {
        return absl::UnimplementedError(absl::StrCat(
            "arith.constant of tensor type requires a DenseElementsAttr: ",
            mlir::debugString(ty)));
      }
      TF_ASSIGN_OR_RETURN(std::string elem_msl,
                          EmitElementType(tensor_ty.getElementType()));
      std::string name = CreateFreshName();
      BindLValue(op.getResult(), name, "thread");
      os_ << elem_msl << " " << name << "[" << tensor_ty.getNumElements()
          << "] = {";
      bool first = true;
      for (mlir::Attribute elt : dense_attr.getValues<mlir::Attribute>()) {
        if (!first) os_ << ", ";
        first = false;
        TF_ASSIGN_OR_RETURN(
            std::string lit,
            FormatConstantLiteral(tensor_ty.getElementType(), elt));
        os_ << lit;
      }
      os_ << "};\n";
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(ty));
    TF_ASSIGN_OR_RETURN(std::string literal,
                        FormatConstantLiteral(ty, op.getValueAttr()));
    std::string name = BindValueName(op.getResult());
    os_ << ty_msl << " " << name << " = " << literal << ";\n";
    return absl::OkStatus();
  }

  absl::Status EmitUbPoison(mlir::ub::PoisonOp op) {
    // MSL has no poison; emit a zero-init declaration. Callers are
    // expected to overwrite before reading.
    mlir::Type ty = op.getResult().getType();
    if (!mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(ty)) {
      return absl::UnimplementedError(
          absl::StrCat("ub.poison of non-scalar type is not supported: ",
                       mlir::debugString(ty)));
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(ty));
    std::string name = BindValueName(op.getResult());
    os_ << ty_msl << " " << name << " = " << ty_msl << "(0);\n";
    return absl::OkStatus();
  }

  // Emits `T name = fn(op0, op1, ...);` — an elementwise math intrinsic that
  // takes the op's operands in order (abs, exp, fma, ...).
  absl::Status EmitMathCall(mlir::Operation* op, absl::string_view fn) {
    if (op->getNumResults() != 1) {
      return absl::InternalError(
          absl::StrCat(op->getName().getStringRef().str(), " has ",
                       op->getNumResults(), " results; expected 1."));
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl,
                        TypeToMSL(op->getResult(0).getType()));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty_msl << " " << name << " = " << fn << "(";
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (i > 0) os_ << ", ";
      TF_ASSIGN_OR_RETURN(std::string arg, GetName(op->getOperand(i)));
      os_ << arg;
    }
    os_ << ");\n";
    return absl::OkStatus();
  }

  // Emits `T name = <op_str>src;` for a unary prefix operator (e.g. negf).
  absl::Status EmitUnaryOp(mlir::Operation* op, absl::string_view op_str) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return absl::InternalError(absl::StrCat(
          op->getName().getStringRef().str(), " has unexpected arity."));
    }
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op->getResult(0).getType()));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty << " " << name << " = " << op_str << src << ";\n";
    return absl::OkStatus();
  }

  // arith.minui / arith.maxui: signless operands compared as unsigned, so
  // cast both to the unsigned MSL type before metal::min / metal::max.
  absl::Status EmitArithUnsignedMinMax(mlir::Operation* op,
                                       absl::string_view fn) {
    TF_ASSIGN_OR_RETURN(std::string u_ty,
                        UnsignedMslType(op->getOperand(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string a, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string b, GetName(op->getOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op->getResult(0).getType()));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty << " " << name << " = " << fn << "(static_cast<" << u_ty << ">("
        << a << "), static_cast<" << u_ty << ">(" << b << "));\n";
    return absl::OkStatus();
  }

  // arith.divui / arith.remui: the MLIR i-types are signless and
  // EmitElementType maps them to signed MSL (int/short/...), so a bare '/' or
  // '%' would be signed. Cast both operands to the unsigned MSL type to get the
  // right semantics; the implicit conversion back to the (signless) result type
  // preserves the bit pattern.
  absl::Status EmitArithUnsignedBinary(mlir::Operation* op,
                                       absl::string_view op_str) {
    TF_ASSIGN_OR_RETURN(std::string u_ty,
                        UnsignedMslType(op->getOperand(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string a, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string b, GetName(op->getOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op->getResult(0).getType()));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty << " " << name << " = (static_cast<" << u_ty << ">(" << a << ") "
        << op_str << " static_cast<" << u_ty << ">(" << b << "));\n";
    return absl::OkStatus();
  }

  // arith.shrui: logical (unsigned) right shift. Cast the shifted value to
  // unsigned so >> zero-fills instead of sign-extending.
  absl::Status EmitArithShrUI(mlir::Operation* op) {
    TF_ASSIGN_OR_RETURN(std::string u_ty,
                        UnsignedMslType(op->getOperand(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string a, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string b, GetName(op->getOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op->getResult(0).getType()));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty << " " << name << " = (static_cast<" << u_ty << ">(" << a
        << ") >> " << b << ");\n";
    return absl::OkStatus();
  }

  // MSL has no cbrt. Expand to copysign(pow(|x|, 1/3), x), matching
  // ElementalIrEmitter::EmitCbrt.
  absl::Status EmitMathCbrt(mlir::Operation* op) {
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op->getResult(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op->getOperand(0)));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty << " " << name << " = metal::copysign(metal::pow(metal::abs("
        << src << "), " << ty << "(1.0 / 3.0)), " << src << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitMathCtpop(mlir::math::CtPopOp op) {
    TF_ASSIGN_OR_RETURN(std::string src_ty,
                        UnsignedMslType(op.getOperand().getType()));
    TF_ASSIGN_OR_RETURN(std::string dst_ty, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getOperand()));
    std::string name = BindValueName(op.getResult());
    os_ << dst_ty << " " << name << " = static_cast<" << dst_ty
        << ">(metal::popcount(static_cast<" << src_ty << ">(" << src
        << ")));\n";
    return absl::OkStatus();
  }

  absl::Status EmitMathCtlz(mlir::math::CountLeadingZerosOp op) {
    TF_ASSIGN_OR_RETURN(std::string src_ty,
                        UnsignedMslType(op.getOperand().getType()));
    TF_ASSIGN_OR_RETURN(std::string dst_ty, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getOperand()));
    std::string name = BindValueName(op.getResult());
    os_ << dst_ty << " " << name << " = static_cast<" << dst_ty
        << ">(metal::clz(static_cast<" << src_ty << ">(" << src << ")));\n";
    return absl::OkStatus();
  }

  // arith.extui zero-extends: reinterpret the source as unsigned (so the
  // widening doesn't sign-extend), then widen to the result type.
  absl::Status EmitArithExtUI(mlir::arith::ExtUIOp op) {
    TF_ASSIGN_OR_RETURN(std::string u_src,
                        UnsignedMslType(op.getIn().getType()));
    TF_ASSIGN_OR_RETURN(std::string dst, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getIn()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = static_cast<" << dst << ">(static_cast<"
        << u_src << ">(" << src << "));\n";
    return absl::OkStatus();
  }

  // arith.index_castui zero-extends integer inputs before converting to index.
  absl::Status EmitArithIndexCastUI(mlir::arith::IndexCastUIOp op) {
    TF_ASSIGN_OR_RETURN(std::string u_src,
                        UnsignedMslType(op.getIn().getType()));
    TF_ASSIGN_OR_RETURN(std::string dst, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getIn()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = static_cast<" << dst << ">(static_cast<"
        << u_src << ">(" << src << "));\n";
    return absl::OkStatus();
  }

  // arith.fptoui: convert to the unsigned result type, then reinterpret into
  // our signed spelling of that type.
  absl::Status EmitArithFPToUI(mlir::arith::FPToUIOp op) {
    TF_ASSIGN_OR_RETURN(std::string u_dst, UnsignedMslType(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string dst, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getIn()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = static_cast<" << dst << ">(static_cast<"
        << u_dst << ">(" << src << "));\n";
    return absl::OkStatus();
  }

  // arith.uitofp: reinterpret the source as unsigned before converting to
  // float (a direct convert would treat it as signed).
  absl::Status EmitArithUIToFP(mlir::arith::UIToFPOp op) {
    TF_ASSIGN_OR_RETURN(std::string u_src,
                        UnsignedMslType(op.getIn().getType()));
    TF_ASSIGN_OR_RETURN(std::string dst, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getIn()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = static_cast<" << dst << ">(static_cast<"
        << u_src << ">(" << src << "));\n";
    return absl::OkStatus();
  }

  // arith.bitcast: same-width bit reinterpretation via MSL's as_type<T>.
  absl::Status EmitArithBitcast(mlir::arith::BitcastOp op) {
    TF_ASSIGN_OR_RETURN(std::string dst, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getIn()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = as_type<" << dst << ">(" << src << ");\n";
    return absl::OkStatus();
  }

  // arith.minimumf / arith.maximumf are IEEE-754-2019 min/max — NaN
  // propagates. MSL's metal::fmin/fmax don't, so we test explicitly.
  absl::Status EmitArithMinMaxF(mlir::Operation* op, bool is_min) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return absl::InternalError(
          absl::StrCat(op->getName().getStringRef().str(),
                       " has unexpected arity: ", op->getNumOperands(),
                       " operand(s), ", op->getNumResults(), " result(s)."));
    }
    mlir::Type ty = op->getResult(0).getType();
    if (!mlir::isa<mlir::FloatType>(ty)) {
      return absl::UnimplementedError(
          absl::StrCat(op->getName().getStringRef().str(),
                       ": only scalar float operands are supported."));
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, GetName(op->getOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(ty));
    std::string name = BindValueName(op->getResult(0));
    absl::string_view fn = is_min ? "metal::fmin" : "metal::fmax";
    os_ << ty_msl << " " << name << " = (metal::isnan(" << lhs
        << ") || metal::isnan(" << rhs << ")) ? metal::numeric_limits<"
        << ty_msl << ">::quiet_NaN() : " << fn << "(" << lhs << ", " << rhs
        << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitArithCmpI(mlir::arith::CmpIOp op) {
    mlir::Type lhs_ty = op.getLhs().getType();
    if (!mlir::isa<mlir::IntegerType, mlir::IndexType>(lhs_ty)) {
      return absl::UnimplementedError(
          "arith.cmpi: only scalar integer/index operands are supported.");
    }
    using P = mlir::arith::CmpIPredicate;
    absl::string_view op_str;
    bool unsigned_cmp = false;
    switch (op.getPredicate()) {
      case P::eq:
        op_str = "==";
        break;
      case P::ne:
        op_str = "!=";
        break;
      case P::slt:
        op_str = "<";
        break;
      case P::sle:
        op_str = "<=";
        break;
      case P::sgt:
        op_str = ">";
        break;
      case P::sge:
        op_str = ">=";
        break;
      case P::ult:
        op_str = "<";
        unsigned_cmp = true;
        break;
      case P::ule:
        op_str = "<=";
        unsigned_cmp = true;
        break;
      case P::ugt:
        op_str = ">";
        unsigned_cmp = true;
        break;
      case P::uge:
        op_str = ">=";
        unsigned_cmp = true;
        break;
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, GetName(op.getLhs()));
    TF_ASSIGN_OR_RETURN(std::string rhs, GetName(op.getRhs()));
    std::string name = BindValueName(op.getResult());
    if (unsigned_cmp) {
      // Cast both operands to the matching unsigned type for the u*
      // predicates (signless ints map to MSL's signed spellings).
      TF_ASSIGN_OR_RETURN(std::string u_msl, UnsignedMslType(lhs_ty));
      os_ << "bool " << name << " = (static_cast<" << u_msl << ">(" << lhs
          << ") " << op_str << " static_cast<" << u_msl << ">(" << rhs
          << "));\n";
    } else {
      os_ << "bool " << name << " = (" << lhs << " " << op_str << " " << rhs
          << ");\n";
    }
    return absl::OkStatus();
  }

  absl::Status EmitArithCmpF(mlir::arith::CmpFOp op) {
    if (!mlir::isa<mlir::FloatType>(op.getLhs().getType())) {
      return absl::UnimplementedError(
          "arith.cmpf: only scalar float operands are supported.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, GetName(op.getLhs()));
    TF_ASSIGN_OR_RETURN(std::string rhs, GetName(op.getRhs()));
    std::string name = BindValueName(op.getResult());
    // "Ordered" (O*) predicates are false if either operand is NaN; the C/MSL
    // relational operators already have that semantics. "Unordered" (U*) are
    // the negation, and ORD/UNO test NaN-ness directly via self-comparison.
    using P = mlir::arith::CmpFPredicate;
    std::string e;
    switch (op.getPredicate()) {
      case P::AlwaysFalse:
        e = "false";
        break;
      case P::OEQ:
        e = absl::StrCat("(", lhs, " == ", rhs, ")");
        break;
      case P::OGT:
        e = absl::StrCat("(", lhs, " > ", rhs, ")");
        break;
      case P::OGE:
        e = absl::StrCat("(", lhs, " >= ", rhs, ")");
        break;
      case P::OLT:
        e = absl::StrCat("(", lhs, " < ", rhs, ")");
        break;
      case P::OLE:
        e = absl::StrCat("(", lhs, " <= ", rhs, ")");
        break;
      case P::ONE:
        e = absl::StrCat("(", lhs, " < ", rhs, " || ", lhs, " > ", rhs, ")");
        break;
      case P::ORD:
        e = absl::StrCat("(", lhs, " == ", lhs, " && ", rhs, " == ", rhs, ")");
        break;
      case P::UEQ:
        e = absl::StrCat("!(", lhs, " < ", rhs, " || ", lhs, " > ", rhs, ")");
        break;
      case P::UGT:
        e = absl::StrCat("!(", lhs, " <= ", rhs, ")");
        break;
      case P::UGE:
        e = absl::StrCat("!(", lhs, " < ", rhs, ")");
        break;
      case P::ULT:
        e = absl::StrCat("!(", lhs, " >= ", rhs, ")");
        break;
      case P::ULE:
        e = absl::StrCat("!(", lhs, " > ", rhs, ")");
        break;
      case P::UNE:
        e = absl::StrCat("(", lhs, " != ", rhs, ")");
        break;
      case P::UNO:
        e = absl::StrCat("(", lhs, " != ", lhs, " || ", rhs, " != ", rhs, ")");
        break;
      case P::AlwaysTrue:
        e = "true";
        break;
    }
    os_ << "bool " << name << " = " << e << ";\n";
    return absl::OkStatus();
  }

  absl::Status EmitArithSelect(mlir::arith::SelectOp op) {
    TF_ASSIGN_OR_RETURN(std::string cond, GetName(op.getCondition()));
    TF_ASSIGN_OR_RETURN(std::string tval, GetName(op.getTrueValue()));
    TF_ASSIGN_OR_RETURN(std::string fval, GetName(op.getFalseValue()));
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(op.getType()));
    std::string name = BindValueName(op.getResult());
    // Scalar i1 condition -> ternary (also valid for a whole-vector result).
    // Element-wise vector condition -> metal::select(false, true, cond).
    if (mlir::isa<mlir::VectorType>(op.getCondition().getType())) {
      os_ << ty_msl << " " << name << " = metal::select(" << fval << ", "
          << tval << ", " << cond << ");\n";
    } else {
      os_ << ty_msl << " " << name << " = (" << cond << " ? " << tval << " : "
          << fval << ");\n";
    }
    return absl::OkStatus();
  }

  // Plain MSL-level type cast. Used by arith.index_cast{,ui}: MSL's `long`
  // (our spelling for `index`) and the platform integer types are all
  // primitive, so the cast is a `static_cast`.
  absl::Status EmitArithCast(mlir::Operation* op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return absl::InternalError(
          absl::StrCat("Cast op ", op->getName().getStringRef().str(),
                       " has unexpected arity: ", op->getNumOperands(),
                       " operand(s), ", op->getNumResults(), " result(s)."));
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl,
                        TypeToMSL(op->getResult(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op->getOperand(0)));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty_msl << " " << name << " = static_cast<" << ty_msl << ">(" << src
        << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitBinary(mlir::Operation* op, absl::string_view op_str) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return absl::InternalError(
          absl::StrCat("Binary op ", op->getName().getStringRef().str(),
                       " has unexpected arity: ", op->getNumOperands(),
                       " operand(s), ", op->getNumResults(), " result(s)."));
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl,
                        TypeToMSL(op->getResult(0).getType()));
    TF_ASSIGN_OR_RETURN(std::string lhs, GetName(op->getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, GetName(op->getOperand(1)));
    std::string name = BindValueName(op->getResult(0));
    os_ << ty_msl << " " << name << " = " << lhs << " " << op_str << " " << rhs
        << ";\n";
    return absl::OkStatus();
  }

  absl::Status EmitTensorExtract(mlir::tensor::ExtractOp op) {
    if (op.getIndices().size() > 1) {
      return absl::UnimplementedError(absl::StrCat(
          "tensor.extract with rank ", op.getIndices().size(),
          " not yet supported by MSL emitter; only rank-0 and rank-1 today."));
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getTensor()));
    std::string idx;
    if (op.getIndices().empty()) {
      idx = "0";
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(op.getIndices().front()));
    }
    std::string name = BindValueName(op.getResult());
    os_ << ty_msl << " " << name << " = " << buf << "[" << idx << "];\n";
    return absl::OkStatus();
  }

  absl::Status HoistSharedAllocations(mlir::func::FuncOp func) {
    llvm::SmallVector<::xla::gpu::AllocateSharedOp> shared;
    func.walk([&](::xla::gpu::AllocateSharedOp op) { shared.push_back(op); });
    for (::xla::gpu::AllocateSharedOp op : shared) {
      auto tensor_ty = mlir::cast<mlir::RankedTensorType>(op.getType());
      TF_ASSIGN_OR_RETURN(std::string elem_msl,
                          EmitElementType(tensor_ty.getElementType()));
      std::string name = CreateFreshName();
      BindLValue(op.getResult(), name, "threadgroup");
      os_ << "threadgroup " << elem_msl << " " << name << "["
          << tensor_ty.getNumElements() << "];\n";
    }
    return absl::OkStatus();
  }

  absl::Status EmitSyncThreads(::xla::gpu::SyncThreadsOp op) {
    os_ << "threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n";
    for (auto [result, operand] :
         llvm::zip(op.getResults(), op.getOperands())) {
      TF_RETURN_IF_ERROR(ForwardLValue(result, operand));
    }
    return absl::OkStatus();
  }

  absl::Status EmitGpuShuffle(mlir::gpu::ShuffleOp op) {
    // gpu.shuffle: (value, offset, width) -> (shuffled_value, valid). Our
    // pipeline pins width to threads_per_warp (= 32, Apple Silicon's SIMD
    // group), so valid is statically true and the MSL simd_shuffle* family
    // applies directly.
    TF_ASSIGN_OR_RETURN(std::string value, GetName(op.getValue()));
    TF_ASSIGN_OR_RETURN(std::string offset, GetName(op.getOffset()));
    TF_ASSIGN_OR_RETURN(std::string ty,
                        TypeToMSL(op.getShuffleResult().getType()));
    absl::string_view fn;
    switch (op.getMode()) {
      case mlir::gpu::ShuffleMode::IDX:
        fn = "metal::simd_shuffle";
        break;
      case mlir::gpu::ShuffleMode::XOR:
        fn = "metal::simd_shuffle_xor";
        break;
      case mlir::gpu::ShuffleMode::UP:
        fn = "metal::simd_shuffle_up";
        break;
      case mlir::gpu::ShuffleMode::DOWN:
        fn = "metal::simd_shuffle_down";
        break;
    }
    std::string shuf = BindValueName(op.getShuffleResult());
    os_ << ty << " " << shuf << " = " << fn << "(" << value << ", " << offset
        << ");\n";
    if (!op.getValid().use_empty()) {
      std::string valid = BindValueName(op.getValid());
      os_ << "bool " << valid << " = true;\n";
    }
    return absl::OkStatus();
  }

  absl::Status EmitDimComponent(mlir::Value result,
                                absl::string_view kernel_attr_name,
                                mlir::gpu::Dimension dim) {
    const char* component = nullptr;
    switch (dim) {
      case mlir::gpu::Dimension::x:
        component = "x";
        break;
      case mlir::gpu::Dimension::y:
        component = "y";
        break;
      case mlir::gpu::Dimension::z:
        component = "z";
        break;
    }
    if (!component) {
      return absl::InternalError("gpu.* op has unknown dimension.");
    }
    TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(result.getType()));
    std::string name = BindValueName(result);
    os_ << ty_msl << " " << name << " = static_cast<" << ty_msl << ">("
        << kernel_attr_name << "." << component << ");\n";
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> VectorTypeToMSL(mlir::VectorType ty) {
    if (ty.getRank() != 1) {
      return absl::UnimplementedError(
          "Multi-rank vector types are not yet supported by the MSL emitter.");
    }
    const int64_t n = ty.getDimSize(0);
    if (n == 1) {
      return EmitElementType(ty.getElementType());
    }
    if (n != 2 && n != 3 && n != 4) {
      return absl::UnimplementedError(
          absl::StrCat("MSL vector width must be 2, 3, or 4; got ", n));
    }
    TF_ASSIGN_OR_RETURN(std::string elem, EmitElementType(ty.getElementType()));
    return absl::StrCat(elem, n);
  }

  absl::StatusOr<std::string> AddrSpaceOf(mlir::Value v) const {
    auto it = values_.find(v);
    if (it == values_.end()) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: no value recorded for ",
          mlir::debugString(v.getType())));
    }
    auto* lv = std::get_if<LValue>(&it->second);
    if (!lv) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: value is a computed rvalue, not a memory location, so "
          "it has no address space: ",
          mlir::debugString(v.getType())));
    }
    return lv->address_space;
  }

  absl::Status EmitVectorTransferRead(mlir::vector::TransferReadOp op) {
    if (op.getIndices().size() != 1 || !op.getPermutationMap().isIdentity()) {
      return absl::UnimplementedError(
          "vector.transfer_read: only rank-1, identity-permutation reads "
          "are supported.");
    }
    TF_ASSIGN_OR_RETURN(std::string vec_ty,
                        VectorTypeToMSL(op.getVectorType()));
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getBase()));
    TF_ASSIGN_OR_RETURN(std::string idx, GetName(op.getIndices().front()));
    TF_ASSIGN_OR_RETURN(std::string space, AddrSpaceOf(op.getBase()));
    std::string name = BindValueName(op.getResult());
    os_ << vec_ty << " " << name << " = *(const " << space << " " << vec_ty
        << "*)(&" << buf << "[" << idx << "]);\n";
    return absl::OkStatus();
  }

  absl::Status EmitVectorTransferWrite(mlir::vector::TransferWriteOp op) {
    if (op.getIndices().size() != 1 || !op.getPermutationMap().isIdentity()) {
      return absl::UnimplementedError(
          "vector.transfer_write: only rank-1, identity-permutation writes "
          "are supported.");
    }
    TF_ASSIGN_OR_RETURN(std::string vec_ty,
                        VectorTypeToMSL(op.getVectorType()));
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getBase()));
    TF_ASSIGN_OR_RETURN(std::string idx, GetName(op.getIndices().front()));
    TF_ASSIGN_OR_RETURN(std::string val, GetName(op.getVector()));
    TF_ASSIGN_OR_RETURN(std::string space, AddrSpaceOf(op.getBase()));
    os_ << "*(" << space << " " << vec_ty << "*)(&" << buf << "[" << idx
        << "]) = " << val << ";\n";
    if (!op.getResults().empty()) {
      TF_RETURN_IF_ERROR(ForwardLValue(op.getResult(), op.getBase()));
    }
    return absl::OkStatus();
  }

  absl::Status EmitVectorBitcast(mlir::vector::BitCastOp op) {
    TF_ASSIGN_OR_RETURN(std::string dst,
                        VectorTypeToMSL(op.getResultVectorType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getSource()));
    std::string name = BindValueName(op.getResult());
    os_ << dst << " " << name << " = as_type<" << dst << ">(" << src
        << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitVectorExtract(mlir::vector::ExtractOp op) {
    auto pos = op.getMixedPosition();
    if (pos.size() != 1) {
      return absl::UnimplementedError(
          "vector.extract: only single-index extracts are supported.");
    }
    if (IsSingleElementVector(op.getSource().getType())) {
      std::optional<int64_t> index = GetStaticVectorIndex(pos);
      if (!index || *index != 0) {
        return absl::UnimplementedError(
            "vector.extract from vector<1xT> requires static index 0.");
      }
      TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op.getType()));
      TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getSource()));
      std::string name = BindValueName(op.getResult());
      os_ << ty << " " << name << " = " << src << ";\n";
      return absl::OkStatus();
    }
    // Index is a compile-time constant or a runtime value; MSL supports
    // dynamic subscripting of its vector types either way.
    std::string idx;
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(pos[0])) {
      idx = std::to_string(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(mlir::cast<mlir::Value>(pos[0])));
    }
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op.getType()));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getSource()));
    std::string name = BindValueName(op.getResult());
    os_ << ty << " " << name << " = " << src << "[" << idx << "];\n";
    return absl::OkStatus();
  }

  absl::Status EmitVectorInsert(mlir::vector::InsertOp op) {
    auto pos = op.getMixedPosition();
    if (pos.size() != 1) {
      return absl::UnimplementedError(
          "vector.insert: only single-index inserts are supported.");
    }
    if (IsSingleElementVector(op.getResult().getType())) {
      std::optional<int64_t> index = GetStaticVectorIndex(pos);
      if (!index || *index != 0) {
        return absl::UnimplementedError(
            "vector.insert into vector<1xT> requires static index 0.");
      }
      TF_ASSIGN_OR_RETURN(std::string ty,
                          VectorTypeToMSL(op.getDestVectorType()));
      TF_ASSIGN_OR_RETURN(std::string val, GetName(op.getValueToStore()));
      std::string name = BindValueName(op.getResult());
      os_ << ty << " " << name << " = " << val << ";\n";
      return absl::OkStatus();
    }
    // Value semantics: copy the destination, then overwrite one component.
    // The index is a compile-time constant or a runtime value; MSL supports
    // dynamic subscripting of its vector types either way.
    std::string idx;
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(pos[0])) {
      idx = std::to_string(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(mlir::cast<mlir::Value>(pos[0])));
    }
    TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(op.getResult().getType()));
    TF_ASSIGN_OR_RETURN(std::string dest, GetName(op.getDest()));
    TF_ASSIGN_OR_RETURN(std::string val, GetName(op.getValueToStore()));
    std::string name = BindValueName(op.getResult());
    os_ << ty << " " << name << " = " << dest << ";\n";
    os_ << name << "[" << idx << "] = " << val << ";\n";
    return absl::OkStatus();
  }

  absl::Status EmitVectorFromElements(mlir::vector::FromElementsOp op) {
    TF_ASSIGN_OR_RETURN(std::string vec_ty,
                        VectorTypeToMSL(op.getResult().getType()));
    std::string name = BindValueName(op.getResult());
    if (IsSingleElementVector(op.getResult().getType())) {
      TF_ASSIGN_OR_RETURN(std::string elt_name, GetName(op.getElements()[0]));
      os_ << vec_ty << " " << name << " = " << elt_name << ";\n";
      return absl::OkStatus();
    }
    os_ << vec_ty << " " << name << " = " << vec_ty << "(";
    bool first = true;
    for (mlir::Value elt : op.getElements()) {
      if (!first) os_ << ", ";
      first = false;
      TF_ASSIGN_OR_RETURN(std::string elt_name, GetName(elt));
      os_ << elt_name;
    }
    os_ << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitVectorBroadcast(mlir::vector::BroadcastOp op) {
    auto dst_ty = mlir::dyn_cast<mlir::VectorType>(op.getResult().getType());
    if (!dst_ty || dst_ty.getRank() != 1) {
      return absl::UnimplementedError(
          "vector.broadcast: only rank-1 result vectors are supported.");
    }
    if (auto src_vec =
            mlir::dyn_cast<mlir::VectorType>(op.getSource().getType())) {
      if (src_vec.getRank() != 1 ||
          src_vec.getDimSize(0) != dst_ty.getDimSize(0)) {
        return absl::UnimplementedError(
          "vector.broadcast: vector-to-vector broadcast must preserve "
          "shape (rank-1, equal width).");
      }
    }
    TF_ASSIGN_OR_RETURN(std::string dst_msl, VectorTypeToMSL(dst_ty));
    TF_ASSIGN_OR_RETURN(std::string src, GetName(op.getSource()));
    std::string name = BindValueName(op.getResult());
    if (IsSingleElementVector(dst_ty)) {
      os_ << dst_msl << " " << name << " = " << src << ";\n";
      return absl::OkStatus();
    }
    // MSL's vector ctor splats a scalar argument and copy-constructs from a
    // same-shape vector.
    os_ << dst_msl << " " << name << " = " << dst_msl << "(" << src << ");\n";
    return absl::OkStatus();
  }

  absl::Status EmitScfFor(mlir::scf::ForOp op) {
    // Two kinds of loop-carried iter_args:
    //   - Tensor/memref: aliased to the init buffer and mutated in place, so
    //     scf.yield hands back that same buffer (no reassignment).
    //   - Scalar/vector: a genuine loop-carried value (e.g. a reduction
    //     accumulator). Declare a mutable variable initialized to the init
    //     and reassign it from the yielded value each iteration.
    for (auto [iter_arg, init] :
         llvm::zip(op.getRegionIterArgs(), op.getInitArgs())) {
      TF_ASSIGN_OR_RETURN(std::string init_name, GetName(init));
      if (mlir::isa<mlir::TensorType, mlir::MemRefType>(iter_arg.getType())) {
        TF_RETURN_IF_ERROR(ForwardLValue(iter_arg, init));
      } else {
        TF_ASSIGN_OR_RETURN(std::string ty, TypeToMSL(iter_arg.getType()));
        std::string var = BindValueName(iter_arg);
        os_ << ty << " " << var << " = " << init_name << ";\n";
      }
    }
    TF_ASSIGN_OR_RETURN(std::string iv_ty,
                        TypeToMSL(op.getInductionVar().getType()));
    std::string iv_name = BindValueName(op.getInductionVar());
    TF_ASSIGN_OR_RETURN(std::string lo, GetName(op.getLowerBound()));
    TF_ASSIGN_OR_RETURN(std::string hi, GetName(op.getUpperBound()));
    TF_ASSIGN_OR_RETURN(std::string st, GetName(op.getStep()));
    os_ << "for (" << iv_ty << " " << iv_name << " = " << lo << "; " << iv_name
        << " < " << hi << "; " << iv_name << " += " << st << ") {\n";
    os_.indent();
    for (mlir::Operation& body_op : op.getBody()->without_terminator()) {
      TF_RETURN_IF_ERROR(EmitOp(&body_op));
    }
    // scf.yield: reassign the mutable (scalar/vector) loop variables; a
    // tensor iter_arg must have been mutated in place, so its yield has to
    // alias the iter_arg.
    auto yield = mlir::cast<mlir::scf::YieldOp>(op.getBody()->getTerminator());
    for (auto [yielded, iter_arg] :
         llvm::zip(yield.getOperands(), op.getRegionIterArgs())) {
      TF_ASSIGN_OR_RETURN(std::string y_name, GetName(yielded));
      TF_ASSIGN_OR_RETURN(std::string a_name, GetName(iter_arg));
      if (y_name == a_name) continue;
      if (mlir::isa<mlir::TensorType, mlir::MemRefType>(iter_arg.getType())) {
        return absl::UnimplementedError(
            "scf.for: tensor iter_arg must be mutated in place; the MSL "
            "emitter does not support swapping the buffer per iteration.");
      }
      os_ << a_name << " = " << y_name << ";\n";
    }
    os_.unindent();
    os_ << "}\n";
    for (auto [result, iter_arg] :
         llvm::zip(op.getResults(), op.getRegionIterArgs())) {
      if (mlir::isa<mlir::TensorType, mlir::MemRefType>(iter_arg.getType())) {
        TF_RETURN_IF_ERROR(ForwardLValue(result, iter_arg));
      } else {
        // Scalar result aliases the iter_arg's mutable loop variable.
        TF_ASSIGN_OR_RETURN(std::string name, GetName(iter_arg));
        values_[result] = RValue{name};
      }
    }
    return absl::OkStatus();
  }

  absl::Status EmitScfIf(mlir::scf::IfOp op) {
    // Scalar/vector results: pre-declare a parent-scope variable, both
    // branches assign via scf.yield. Tensor/memref results: alias to the
    // function-scope buffer both branches must yield.
    std::vector<std::string> scalar_names(op.getNumResults());
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      mlir::Value result = op.getResult(i);
      if (mlir::isa<mlir::TensorType, mlir::MemRefType>(result.getType())) {
        continue;  // Resolved as an alias after both branches emit.
      }
      TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(result.getType()));
      std::string name = BindValueName(result);
      os_ << ty_msl << " " << name << ";\n";
      scalar_names[i] = std::move(name);
    }

    auto emit_branch = [&](mlir::Region& region) -> absl::Status {
      for (mlir::Operation& body_op : region.front().without_terminator()) {
        TF_RETURN_IF_ERROR(EmitOp(&body_op));
      }
      if (op.getNumResults() == 0) return absl::OkStatus();
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(region.front().getTerminator());
      for (size_t i = 0; i < op.getNumResults(); ++i) {
        if (scalar_names[i].empty()) continue;  // tensor/memref result.
        TF_ASSIGN_OR_RETURN(std::string y_name, GetName(yield.getOperand(i)));
        os_ << scalar_names[i] << " = " << y_name << ";\n";
      }
      return absl::OkStatus();
    };

    TF_ASSIGN_OR_RETURN(std::string cond, GetName(op.getCondition()));
    os_ << "if (" << cond << ") {\n";
    os_.indent();
    TF_RETURN_IF_ERROR(emit_branch(op.getThenRegion()));
    os_.unindent();
    if (!op.getElseRegion().empty()) {
      os_ << "} else {\n";
      os_.indent();
      TF_RETURN_IF_ERROR(emit_branch(op.getElseRegion()));
      os_.unindent();
    }
    os_ << "}\n";

    // Buffer-result aliasing. MLIR's scf.if verifier guarantees both
    // regions exist when there are results.
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      mlir::Value result = op.getResult(i);
      if (!mlir::isa<mlir::TensorType, mlir::MemRefType>(result.getType())) {
        continue;
      }
      auto then_yield = mlir::cast<mlir::scf::YieldOp>(
          op.getThenRegion().front().getTerminator());
      auto else_yield = mlir::cast<mlir::scf::YieldOp>(
          op.getElseRegion().front().getTerminator());
      mlir::Value then_v = then_yield.getOperand(i);
      mlir::Value else_v = else_yield.getOperand(i);
      TF_ASSIGN_OR_RETURN(std::string then_name, GetName(then_v));
      TF_ASSIGN_OR_RETURN(std::string else_name, GetName(else_v));
      if (then_name != else_name) {
        return absl::UnimplementedError(
            "scf.if with tensor/memref results: branches yield different "
            "buffers; MSL emitter only supports both branches yielding the "
            "same buffer (the in-place tensor.insert pattern).");
      }
      TF_RETURN_IF_ERROR(ForwardLValue(result, then_v));
    }
    return absl::OkStatus();
  }

  absl::Status EmitScfIndexSwitch(mlir::scf::IndexSwitchOp op) {
    const unsigned num_results = op->getNumResults();
    std::vector<std::string> scalar_names(num_results);
    for (unsigned i = 0; i < num_results; ++i) {
      mlir::Value result = op->getResult(i);
      if (mlir::isa<mlir::TensorType, mlir::MemRefType>(result.getType())) {
        continue;  // Resolved as an alias after every arm emits.
      }
      TF_ASSIGN_OR_RETURN(std::string ty_msl, TypeToMSL(result.getType()));
      std::string name = BindValueName(result);
      os_ << ty_msl << " " << name << ";\n";
      scalar_names[i] = std::move(name);
    }

    auto emit_region = [&](mlir::Region& region) -> absl::Status {
      for (mlir::Operation& body_op : region.front().without_terminator()) {
        TF_RETURN_IF_ERROR(EmitOp(&body_op));
      }
      if (num_results == 0) return absl::OkStatus();
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(region.front().getTerminator());
      for (unsigned i = 0; i < num_results; ++i) {
        if (scalar_names[i].empty()) continue;  // tensor/memref result.
        TF_ASSIGN_OR_RETURN(std::string y_name, GetName(yield.getOperand(i)));
        os_ << scalar_names[i] << " = " << y_name << ";\n";
      }
      return absl::OkStatus();
    };

    TF_ASSIGN_OR_RETURN(std::string arg, GetName(op.getArg()));
    os_ << "switch (" << arg << ") {\n";
    os_.indent();
    for (auto [case_value, case_region] :
         llvm::zip(op.getCases(), op.getCaseRegions())) {
      os_ << "case " << case_value << ": {\n";
      os_.indent();
      TF_RETURN_IF_ERROR(emit_region(case_region));
      os_ << "break;\n";
      os_.unindent();
      os_ << "}\n";
    }
    os_ << "default: {\n";
    os_.indent();
    TF_RETURN_IF_ERROR(emit_region(op.getDefaultRegion()));
    os_ << "break;\n";
    os_.unindent();
    os_ << "}\n";
    os_.unindent();
    os_ << "}\n";

    for (unsigned i = 0; i < num_results; ++i) {
      mlir::Value result = op->getResult(i);
      if (!mlir::isa<mlir::TensorType, mlir::MemRefType>(result.getType())) {
        continue;
      }

      mlir::Value common_value;
      std::string common_name;
      for (mlir::Region& region : op->getRegions()) {
        auto yield =
            mlir::cast<mlir::scf::YieldOp>(region.front().getTerminator());
        mlir::Value yielded = yield.getOperand(i);
        TF_ASSIGN_OR_RETURN(std::string name, GetName(yielded));
        if (!common_value) {
          common_value = yielded;
          common_name = std::move(name);
          continue;
        }
        if (name != common_name) {
          return absl::UnimplementedError(
              "scf.index_switch with tensor/memref results: arms yield "
              "different buffers; MSL emitter only supports every arm "
              "yielding the same buffer for each result.");
        }
      }
      TF_RETURN_IF_ERROR(ForwardLValue(result, common_value));
    }
    return absl::OkStatus();
  }

  absl::Status EmitTensorInsert(mlir::tensor::InsertOp op) {
    if (op.getIndices().size() > 1) {
      return absl::UnimplementedError(absl::StrCat(
          "tensor.insert with rank ", op.getIndices().size(),
          " not yet supported by MSL emitter; only rank-0 and rank-1 today."));
    }
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getDest()));
    std::string idx;
    if (op.getIndices().empty()) {
      idx = "0";
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(op.getIndices().front()));
    }
    TF_ASSIGN_OR_RETURN(std::string val, GetName(op.getScalar()));
    os_ << buf << "[" << idx << "] = " << val << ";\n";
    // Result aliases dest; subsequent users access the same buffer.
    TF_RETURN_IF_ERROR(ForwardLValue(op.getResult(), op.getDest()));
    return absl::OkStatus();
  }

  struct AtomicStorage {
    std::string atomic_ty;
    std::string storage_ty;
    bool bitcast_through_storage = false;
  };

  absl::StatusOr<AtomicStorage> GetAtomicStorage(mlir::Type element_ty) {
    if (element_ty.isF32()) {
      return AtomicStorage{/*atomic_ty=*/"atomic_uint",
                           /*storage_ty=*/"uint",
                           /*bitcast_through_storage=*/true};
    }
    if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(element_ty)) {
      if (int_ty.getWidth() != 32) {
        return absl::UnimplementedError(absl::StrCat(
            "xla.atomic_rmw: only 32-bit integer atomics are supported by "
            "the MSL emitter today; got ",
            mlir::debugString(element_ty)));
      }
      return AtomicStorage{/*atomic_ty=*/int_ty.isUnsigned() ? "atomic_uint"
                                                             : "atomic_int",
                           /*storage_ty=*/int_ty.isUnsigned() ? "uint" : "int",
                           /*bitcast_through_storage=*/false};
    }
    return absl::UnimplementedError(absl::StrCat(
        "xla.atomic_rmw: unsupported element type for MSL CAS lowering: ",
        mlir::debugString(element_ty)));
  }

  absl::Status EmitSub32BitAtomicRMW(::xla::AtomicRMWOp op, int element_width,
                                     bool bitcast_through_storage) {
    if (element_width != 8 && element_width != 16) {
      return absl::UnimplementedError(absl::StrCat(
          "xla.atomic_rmw: sub-32-bit atomics only support 8/16-bit values "
          "today; "
          "got i",
          element_width));
    }

    auto tensor_ty = mlir::cast<mlir::RankedTensorType>(
        op.getInput().getType());
    mlir::Type element_ty = tensor_ty.getElementType();
    TF_ASSIGN_OR_RETURN(std::string element_msl, TypeToMSL(element_ty));
    const std::string storage_msl = element_width == 8 ? "uchar" : "ushort";
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getInput()));
    TF_ASSIGN_OR_RETURN(std::string space, AddrSpaceOf(op.getInput()));

    std::string idx;
    if (op.getIndices().empty()) {
      idx = "0";
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(op.getIndices().front()));
    }

    const int bytes_per_element = element_width / 8;
    const uint32_t element_mask = (uint32_t{1} << element_width) - 1;

    std::string element_index = CreateFreshName();
    os_ << "ulong " << element_index << " = static_cast<ulong>(" << idx
        << ");\n";
    std::string byte_offset = CreateFreshName();
    os_ << "ulong " << byte_offset << " = " << element_index;
    if (bytes_per_element != 1) {
      os_ << " * " << bytes_per_element << "ul";
    }
    os_ << ";\n";
    std::string ptr = CreateFreshName();
    os_ << space << " atomic_uint* " << ptr << " = ((" << space
        << " atomic_uint*)" << buf << ") + (" << byte_offset << " >> 2);\n";
    std::string shift = CreateFreshName();
    os_ << "uint " << shift << " = static_cast<uint>((" << byte_offset
        << " & 3ul) * 8ul);\n";
    std::string field_mask = CreateFreshName();
    os_ << "uint " << field_mask << " = " << element_mask << "u << " << shift
        << ";\n";
    std::string expected = CreateFreshName();
    os_ << "uint " << expected << " = atomic_load_explicit(" << ptr
        << ", memory_order_relaxed);\n";
    std::string success = CreateFreshName();
    os_ << "bool " << success << " = false;\n";
    os_ << "do {\n";
    os_.indent();

    std::string current = CreateFreshName();
    if (bitcast_through_storage) {
      std::string current_storage = CreateFreshName();
      os_ << storage_msl << " " << current_storage << " = static_cast<"
          << storage_msl << ">((" << expected << " >> " << shift << ") & "
          << element_mask << "u);\n";
      os_ << element_msl << " " << current << " = as_type<" << element_msl
          << ">(" << current_storage << ");\n";
    } else {
      os_ << element_msl << " " << current << " = static_cast<" << element_msl
          << ">((" << expected << " >> " << shift << ") & " << element_mask
          << "u);\n";
    }
    values_[op.getCurrentValue()] = RValue{current};

    for (mlir::Operation& body_op : op.getBody()->without_terminator()) {
      TF_RETURN_IF_ERROR(EmitOp(&body_op));
    }
    auto yield = mlir::dyn_cast<::xla::YieldOp>(op.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1) {
      return absl::InvalidArgumentError(
          "xla.atomic_rmw expected a single-value xla.yield terminator.");
    }
    TF_ASSIGN_OR_RETURN(std::string yielded, GetName(yield.getOperand(0)));

    std::string desired_field = CreateFreshName();
    if (bitcast_through_storage) {
      std::string desired_storage = CreateFreshName();
      os_ << storage_msl << " " << desired_storage << " = as_type<"
          << storage_msl << ">(" << yielded << ");\n";
      os_ << "uint " << desired_field << " = (static_cast<uint>("
          << desired_storage << ") & " << element_mask << "u) << " << shift
          << ";\n";
    } else {
      os_ << "uint " << desired_field << " = (static_cast<uint>(" << yielded
          << ") & " << element_mask << "u) << " << shift << ";\n";
    }
    std::string desired = CreateFreshName();
    os_ << "uint " << desired << " = (" << expected << " & ~" << field_mask
        << ") | " << desired_field << ";\n";
    os_ << success << " = atomic_compare_exchange_weak_explicit(" << ptr
        << ", &" << expected << ", " << desired
        << ", memory_order_relaxed, memory_order_relaxed);\n";
    os_.unindent();
    os_ << "} while (!" << success << ");\n";

    TF_RETURN_IF_ERROR(ForwardLValue(op.getResult(), op.getInput()));
    return absl::OkStatus();
  }

  absl::Status EmitXlaAtomicRMW(::xla::AtomicRMWOp op) {
    auto tensor_ty = mlir::dyn_cast<mlir::RankedTensorType>(
        op.getInput().getType());
    if (!tensor_ty) {
      return absl::InvalidArgumentError(absl::StrCat(
          "xla.atomic_rmw expected a ranked tensor input, got ",
          mlir::debugString(op.getInput().getType())));
    }
    if (op.getIndices().size() > 1) {
      return absl::UnimplementedError(absl::StrCat(
          "xla.atomic_rmw with rank ", op.getIndices().size(),
          " not yet supported by MSL emitter; only rank-0 and rank-1 today."));
    }

    mlir::Type element_ty = tensor_ty.getElementType();
    if (mlir::isa<mlir::VectorType>(op.getCurrentValue().getType())) {
      return absl::UnimplementedError(
          "xla.atomic_rmw: vector atomics are not yet supported by the MSL "
          "emitter.");
    }
    if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(element_ty);
        int_ty && int_ty.getWidth() < 32) {
      return EmitSub32BitAtomicRMW(op, int_ty.getWidth(),
                                   /*bitcast_through_storage=*/false);
    }
    if (element_ty.isF16()) {
      return EmitSub32BitAtomicRMW(op, /*element_width=*/16,
                                   /*bitcast_through_storage=*/true);
    }
    TF_ASSIGN_OR_RETURN(AtomicStorage atomic_storage,
                        GetAtomicStorage(element_ty));
    TF_ASSIGN_OR_RETURN(std::string element_msl, TypeToMSL(element_ty));
    TF_ASSIGN_OR_RETURN(std::string buf, GetName(op.getInput()));
    TF_ASSIGN_OR_RETURN(std::string space, AddrSpaceOf(op.getInput()));

    std::string idx;
    if (op.getIndices().empty()) {
      idx = "0";
    } else {
      TF_ASSIGN_OR_RETURN(idx, GetName(op.getIndices().front()));
    }

    std::string ptr = CreateFreshName();
    os_ << space << " " << atomic_storage.atomic_ty << "* " << ptr << " = ("
        << space << " " << atomic_storage.atomic_ty << "*)(&" << buf << "["
        << idx << "]);\n";

    std::string expected = CreateFreshName();
    os_ << atomic_storage.storage_ty << " " << expected
        << " = atomic_load_explicit(" << ptr << ", memory_order_relaxed);\n";
    std::string success = CreateFreshName();
    os_ << "bool " << success << " = false;\n";
    os_ << "do {\n";
    os_.indent();

    std::string current = CreateFreshName();
    if (atomic_storage.bitcast_through_storage) {
      os_ << element_msl << " " << current << " = as_type<" << element_msl
          << ">(" << expected << ");\n";
    } else {
      os_ << element_msl << " " << current << " = static_cast<" << element_msl
          << ">(" << expected << ");\n";
    }
    values_[op.getCurrentValue()] = RValue{current};

    for (mlir::Operation& body_op : op.getBody()->without_terminator()) {
      TF_RETURN_IF_ERROR(EmitOp(&body_op));
    }
    auto yield = mlir::dyn_cast<::xla::YieldOp>(op.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1) {
      return absl::InvalidArgumentError(
          "xla.atomic_rmw expected a single-value xla.yield terminator.");
    }
    TF_ASSIGN_OR_RETURN(std::string yielded, GetName(yield.getOperand(0)));

    std::string desired = CreateFreshName();
    if (atomic_storage.bitcast_through_storage) {
      os_ << atomic_storage.storage_ty << " " << desired << " = as_type<"
          << atomic_storage.storage_ty << ">(" << yielded << ");\n";
    } else {
      os_ << atomic_storage.storage_ty << " " << desired << " = static_cast<"
          << atomic_storage.storage_ty << ">(" << yielded << ");\n";
    }
    os_ << success << " = atomic_compare_exchange_weak_explicit(" << ptr
        << ", &" << expected << ", " << desired
        << ", memory_order_relaxed, memory_order_relaxed);\n";
    os_.unindent();
    os_ << "} while (!" << success << ");\n";

    TF_RETURN_IF_ERROR(ForwardLValue(op.getResult(), op.getInput()));
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> GetName(mlir::Value v) const {
    auto it = values_.find(v);
    if (it == values_.end()) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: SSA value has no name (use before def or unmapped "
          "operand). Type: ",
          mlir::debugString(v.getType())));
    }
    if (auto* lv = std::get_if<LValue>(&it->second)) return lv->name;
    return std::get<RValue>(it->second).name;
  }

  std::string BindValueName(mlir::Value v) {
    std::string name = CreateFreshName();
    values_[v] = RValue{name};
    return name;
  }

  // Records `v` as an lvalue: a memory location with MSL identifier `name` in
  // address space `address_space`.
  void BindLValue(mlir::Value v, std::string name, std::string address_space) {
    values_[v] = LValue{std::move(name), std::move(address_space)};
  }

  // Forwards an lvalue (name + address space, inseparably) from `from` to `to`,
  // for ops that thread a buffer/tile through unchanged (scf.if/for,
  // sync_threads, tensor.insert, atomic results). Errors loudly if `from` is a
  // computed rvalue rather than a memory location.
  absl::Status ForwardLValue(mlir::Value to, mlir::Value from) {
    auto it = values_.find(from);
    if (it == values_.end()) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: cannot forward an unrecorded value: ",
          mlir::debugString(from.getType())));
    }
    if (!std::holds_alternative<LValue>(it->second)) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: expected a memory lvalue to forward but got a "
          "computed rvalue: ",
          mlir::debugString(from.getType())));
    }
    // Copy before assigning: inserting `to` can rehash values_ and invalidate
    // a reference into it (a use-after-free of the source record otherwise).
    LValue lv = std::get<LValue>(it->second);
    values_[to] = std::move(lv);
    return absl::OkStatus();
  }

  std::string CreateFreshName() { return absl::StrCat("v", next_id_++); }

  std::string EmittedFunctionName(mlir::func::FuncOp func) const {
    return EmittedFunctionName(func.getName());
  }

  std::string EmittedFunctionName(llvm::StringRef name) const {
    auto it = emitted_function_names_.find(name);
    CHECK(it != emitted_function_names_.end())
        << "Missing emitted function name for " << name.str();
    return it->second;
  }

  std::string EmittedFunctionResultTypeName(mlir::func::FuncOp func) const {
    return EmittedFunctionResultTypeName(func.getName());
  }

  std::string EmittedFunctionResultTypeName(llvm::StringRef name) const {
    auto it = emitted_result_type_names_.find(name);
    CHECK(it != emitted_result_type_names_.end())
        << "Missing emitted result type name for " << name.str();
    return it->second;
  }

  mlir::raw_indented_ostream& os_;
  // An emitted SSA value is either an lvalue — a memory location (buffer, tile,
  // or local array) with an MSL identifier and an address space — or an rvalue:
  // a computed scalar/vector with just an identifier. Holding the name and
  // address space in one record means an lvalue forwarded through
  // scf.if/sync_threads/atomic results can never lose its space (see
  // ForwardLValue), which is the class of bug a parallel name/space map invites.
  struct LValue {
    std::string name;
    std::string address_space;  // "device" | "threadgroup" | "thread"
  };
  struct RValue {
    std::string name;
  };
  using EmittedValue = std::variant<LValue, RValue>;
  llvm::DenseMap<mlir::Value, EmittedValue> values_;
  unsigned next_id_ = 0;
  // True while emitting the entry kernel (void; results land in buffers) and
  // false while emitting a device function (func.return yields a value).
  bool in_entry_ = false;
  llvm::StringMap<std::string> emitted_function_names_;
  llvm::StringMap<std::string> emitted_result_type_names_;
};

}  // namespace

absl::StatusOr<MslKernelSource> TranslateToMSL(mlir::ModuleOp module) {
  NameUniquer function_name_uniquer;
  return TranslateToMSL(module, &function_name_uniquer);
}

absl::StatusOr<MslKernelSource> TranslateToMSL(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer) {
  mlir::func::FuncOp entry;
  std::vector<mlir::func::FuncOp> helpers;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!func->hasAttr(std::string(kEntryAttrName))) {
      helpers.push_back(func);
      continue;
    }
    if (entry) {
      return absl::InvalidArgumentError(
          "Module has more than one func.func marked with xla.entry; "
          "TranslateToMSL expects exactly one.");
    }
    entry = func;
  }
  if (!entry) {
    return absl::InvalidArgumentError(
        "Module has no entry func.func (none carry the xla.entry attribute).");
  }

  std::string output;
  llvm::raw_string_ostream raw(output);
  mlir::raw_indented_ostream os(raw);

  os << "#include <metal_stdlib>\n";
  os << "using namespace metal;\n";
  os << "\n";

  llvm::StringMap<std::string> emitted_function_names;
  const std::string entry_point =
      GetUniqueMslFunctionName(entry.getName(), function_name_uniquer);
  emitted_function_names[entry.getName()] = entry_point;
  for (mlir::func::FuncOp helper : helpers) {
    emitted_function_names[helper.getName()] = GetUniqueMslHelperName(
        entry.getName(), helper.getName(), function_name_uniquer);
  }
  llvm::StringMap<std::string> emitted_result_type_names;
  bool has_multi_result_helper = false;
  for (mlir::func::FuncOp helper : helpers) {
    if (helper.getFunctionType().getResults().size() < 2) continue;
    has_multi_result_helper = true;
    emitted_result_type_names[helper.getName()] = GetUniqueMslFunctionName(
        absl::StrCat(emitted_function_names[helper.getName()], "_result"),
        function_name_uniquer);
  }

  MslEmitter emitter(os, std::move(emitted_function_names),
                     std::move(emitted_result_type_names));
  // Emit result structs and forward declarations first so device functions may
  // call one another regardless of emission order.
  for (mlir::func::FuncOp helper : helpers) {
    TF_RETURN_IF_ERROR(emitter.EmitDeviceFunctionResultType(helper));
  }
  if (has_multi_result_helper) os << "\n";
  for (mlir::func::FuncOp helper : helpers) {
    TF_RETURN_IF_ERROR(emitter.EmitDeviceFunctionDecl(helper));
  }
  if (!helpers.empty()) os << "\n";
  for (mlir::func::FuncOp helper : helpers) {
    TF_RETURN_IF_ERROR(emitter.EmitDeviceFunction(helper));
    os << "\n";
  }
  TF_RETURN_IF_ERROR(emitter.EmitFunction(entry));

  raw.flush();
  return MslKernelSource(std::move(output), entry_point);
}

}  // namespace metal
}  // namespace xla
