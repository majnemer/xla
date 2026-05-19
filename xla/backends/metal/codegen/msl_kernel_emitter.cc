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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
#include "xla/backends/metal/codegen/translate_to_msl.h"

namespace xla {
namespace metal {

absl::StatusOr<MslKernelSource> EmitMslKernel(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(CreateConvertComplexToArithMathPass());
  pm.addPass(CreateExpandFloatOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(module))) {
    return absl::InvalidArgumentError(
        "Pre-translation pass pipeline failed on the input module.");
  }
  return TranslateToMSL(module);
}

}  // namespace metal
}  // namespace xla
