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

#ifndef XLA_BACKENDS_METAL_COMPILER_H_
#define XLA_BACKENDS_METAL_COMPILER_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace metal {

// XLA compiler for the Apple Metal backend.
//
// Inherits xla::gpu::GpuCompiler (the post-split, LLVM-free base) and reuses
// the shared GPU machinery wholesale — RunHloPasses, RunBackend, Compile,
// CompileAheadOfTime, scheduling, alias info. The only thing Metal owns is
// CompileToBackendResult, where MSL kernel emission will eventually live.
class MetalCompiler : public xla::gpu::GpuCompiler {
 public:
  MetalCompiler();
  ~MetalCompiler() override = default;

 protected:
  void AddGemmRewriteCustomCallPasses(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      se::GpuComputeCapability gpu_version,
      const se::SemanticVersion& toolkit_version) override;

  // Produces the GpuExecutable. Until MSL emission for compute ops lands,
  // only HLO modules whose entry computation is parameter-only are
  // supported; compute opcodes are rejected with Unimplemented up front.
  absl::StatusOr<std::unique_ptr<xla::gpu::GpuExecutable>>
  CompileToBackendResult(std::unique_ptr<HloModule> module,
                         const GpuTopology& gpu_topology,
                         const CompileOptions& options,
                         se::StreamExecutor* absl_nullable stream_exec)
      override;

 private:
  MetalCompiler(const MetalCompiler&) = delete;
  MetalCompiler& operator=(const MetalCompiler&) = delete;
};

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_COMPILER_H_
