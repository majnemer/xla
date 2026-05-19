/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/llvm_irgen_test_base.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_llvm_compiler.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Executable>>
LlvmIrGenTestBase::CompileToExecutable(std::unique_ptr<HloModule> hlo_module,
                                       bool run_optimization_passes) {
  if (run_optimization_passes) {
    TF_ASSIGN_OR_RETURN(hlo_module, backend().compiler()->RunHloPasses(
                                        std::move(hlo_module),
                                        backend().default_stream_executor(),
                                        /*device_allocator=*/nullptr));
  }
  return backend().compiler()->RunBackend(std::move(hlo_module),
                                          backend().default_stream_executor(),
                                          /*device_allocator=*/nullptr);
}

void LlvmIrGenTestBase::SetIrHook(bool match_optimized_ir) {
  using std::placeholders::_1;
  auto hook = std::bind(&LlvmIrGenTestBase::IrHook, this, _1);

  Compiler* base_compiler = backend().compiler();
  if (auto* llvm_compiler = dynamic_cast<LLVMCompiler*>(base_compiler)) {
    if (match_optimized_ir) {
      llvm_compiler->SetPostOptimizationHook(hook);
    } else {
      llvm_compiler->SetPreOptimizationHook(hook);
    }
    return;
  }
  auto* gpu_llvm_compiler =
      dynamic_cast<gpu::GpuLLVMCompiler*>(base_compiler);
  CHECK_NE(gpu_llvm_compiler, nullptr);
  if (match_optimized_ir) {
    gpu_llvm_compiler->SetPostOptimizationHook(hook);
  } else {
    gpu_llvm_compiler->SetPreOptimizationHook(hook);
  }
}

void LlvmIrGenTestBase::ResetIrHook() {
  Compiler* base_compiler = backend().compiler();
  if (auto* llvm_compiler = dynamic_cast<LLVMCompiler*>(base_compiler)) {
    llvm_compiler->RemovePreOptimizationHook();
    llvm_compiler->RemovePostOptimizationHook();
    return;
  }
  auto* gpu_llvm_compiler =
      dynamic_cast<gpu::GpuLLVMCompiler*>(base_compiler);
  CHECK_NE(gpu_llvm_compiler, nullptr);
  gpu_llvm_compiler->RemovePreOptimizationHook();
  gpu_llvm_compiler->RemovePostOptimizationHook();
}

void LlvmIrGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const std::string& pattern,
    bool match_optimized_ir, bool run_optimization_passes) {
  SetIrHook(match_optimized_ir);
  absl::Status status =
      CompileToExecutable(std::move(hlo_module), run_optimization_passes)
          .status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::CompileAndVerifyIr(const std::string& hlo_text,
                                           const std::string& expected_llvm_ir,
                                           bool match_optimized_ir,
                                           bool run_optimization_passes) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, match_optimized_ir,
                     run_optimization_passes);
}

absl::Status LlvmIrGenTestBase::IrHook(const llvm::Module& module) {
  ir_ += llvm_ir::DumpToString(&module);
  return absl::OkStatus();
}

}  // namespace xla
