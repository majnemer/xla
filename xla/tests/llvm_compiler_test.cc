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

#include "xla/service/llvm_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_llvm_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using LLVMCompilerTest = HloTestBase;

const char* const kHloText = R"(
HloModule Add

ENTRY main  {
  constant.0 = f32[] constant(42.0)
  constant.1 = f32[] constant(43.0)
  ROOT add.0 = f32[] add(constant.0, constant.1)
}
)";

TEST_F(LLVMCompilerTest, HooksTest) {
  int pre_opt_hook_call_count = 0;
  int post_opt_hook_call_count = 0;

  auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module&) {
    ++pre_opt_hook_call_count;
    return absl::OkStatus();
  };
  auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module&) {
    ++post_opt_hook_call_count;
    return absl::OkStatus();
  };

  // Create HLO module. Note this module needs to consist of at least one
  // instruction that is compiled using LLVM (e.g. for CPU thunks runtime it is
  // 'add' instruction), otherwise the hooks are never called.
  auto hlo_module = ParseAndReturnVerifiedModule(kHloText).value();

  // Both CPU (LLVMCompiler) and GPU (gpu::GpuLLVMCompiler) expose the same
  // pre/post-optimization hook API, but no longer share a base class, so this
  // test runtime-selects which one is in use.
  Compiler* base_compiler = backend().compiler();
  auto* llvm_compiler = dynamic_cast<xla::LLVMCompiler*>(base_compiler);
  auto* gpu_llvm_compiler =
      dynamic_cast<xla::gpu::GpuLLVMCompiler*>(base_compiler);
  ASSERT_TRUE(llvm_compiler != nullptr || gpu_llvm_compiler != nullptr);
  if (llvm_compiler != nullptr) {
    llvm_compiler->SetPreOptimizationHook(pre_opt_hook);
    llvm_compiler->SetPostOptimizationHook(post_opt_hook);
  } else {
    gpu_llvm_compiler->SetPreOptimizationHook(pre_opt_hook);
    gpu_llvm_compiler->SetPostOptimizationHook(post_opt_hook);
  }

  ASSERT_TRUE(base_compiler
                  ->RunBackend(std::move(hlo_module),
                               backend().default_stream_executor(),
                               /*device_allocator=*/nullptr)
                  .ok());

  // Test that hooks were called.
  EXPECT_LE(1, pre_opt_hook_call_count);
  EXPECT_LE(1, post_opt_hook_call_count);
}

}  // namespace
}  // namespace xla
