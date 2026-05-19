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

#include "xla/tests/codegen_utils.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
    Compiler* compiler, const Compiler::CompileOptions& compile_options,
    std::unique_ptr<HloModule> hlo_module, bool run_optimization_passes) {
  if (run_optimization_passes) {
    TF_ASSIGN_OR_RETURN(hlo_module, compiler->RunHloPasses(
                                        std::move(hlo_module),
                                        /*executor=*/nullptr, compile_options));
  }
  return compiler->RunBackend(std::move(hlo_module), /*executor=*/nullptr,
                              compile_options);
}

}  // namespace xla
