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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
#include "xla/backends/metal/codegen/translate_to_msl.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/ir_printing.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/name_uniquer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace metal {
namespace {

absl::Status RunPreTranslationPipeline(
    mlir::ModuleOp module, const HloModule* hlo_module,
    absl::string_view entry_function_name) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(CreateConvertComplexToArithMathPass());
  pm.addPass(CreateExpandFloatOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (hlo_module != nullptr) {
    std::string dump_kernel_name =
        absl::StrCat(entry_function_name, ".msl-pretranslation");
    EnableIRPrintingIfRequested(pm, module.getContext(), *hlo_module,
                                dump_kernel_name, "mlir-fusion");
  }
  if (mlir::failed(pm.run(module))) {
    return absl::InvalidArgumentError(
        "Pre-translation pass pipeline failed on the input module.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RunMetalLoweringPipeline(
    mlir::ModuleOp module, const stream_executor::DeviceDescription& device,
    int max_unroll_factor, const HloModule& hlo_module,
    absl::string_view entry_function_name, absl::string_view dump_category) {
  mlir::PassManager pm(module.getContext());
  gpu::AddLoopTransformationPasses(pm, device, max_unroll_factor,
                                   /*max_vector_elements=*/4);
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateConvertPureCallOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(gpu::CreateConvertIndexTypePass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(CreateConvertComplexToArithMathPass());
  pm.addPass(emitters::CreateExpandFloatOpsPass());
  pm.addPass(CreateExpandFloatOpsPass());
  pm.addPass(CreateLowerSubByteStoragePass());
  pm.addPass(CreateLowerFloatStoragePass());
  pm.addPass(mlir::createLowerAffinePass());
  std::string dump_kernel_name =
      absl::StrCat(entry_function_name, ".metal-lowering");
  EnableIRPrintingIfRequested(pm, module.getContext(), hlo_module,
                              dump_kernel_name, dump_category);
  if (mlir::failed(pm.run(module))) {
    return absl::InternalError(absl::StrCat(
        "Metal MLIR lowering pipeline failed on '", entry_function_name, "'."));
  }
  return absl::OkStatus();
}

absl::StatusOr<MslKernelSource> EmitMslKernel(mlir::ModuleOp module) {
  NameUniquer function_name_uniquer;
  return EmitMslKernel(module, &function_name_uniquer);
}

absl::StatusOr<MslKernelSource> EmitMslKernel(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer) {
  TF_RETURN_IF_ERROR(RunPreTranslationPipeline(
      module, /*hlo_module=*/nullptr, /*entry_function_name=*/""));
  return TranslateToMSL(module, function_name_uniquer);
}

absl::StatusOr<MslKernelSource> EmitMslKernel(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer,
    const HloModule& hlo_module, absl::string_view entry_function_name) {
  TF_RETURN_IF_ERROR(
      RunPreTranslationPipeline(module, &hlo_module, entry_function_name));
  return TranslateToMSL(module, function_name_uniquer);
}

}  // namespace metal
}  // namespace xla
