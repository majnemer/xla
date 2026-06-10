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

#ifndef XLA_SERVICE_GPU_GPU_LLVM_COMPILER_H_
#define XLA_SERVICE_GPU_GPU_LLVM_COMPILER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Intermediate base for LLVM-flavored GPU compilers (CUDA / ROCm / SYCL).
// Holds everything LLVM-specific that GpuCompiler used to own: the LLVM
// target_triple / data_layout, the LLVM-binary compile pipeline
// (CompileSingleModule / CompileToBackendResult), and AOT serialization
// that round-trips through LLVM binaries.
//
// Inherits both GpuCompiler (the GPU pipeline) and LLVMCompiler (the
// IR-inspection hook surface); both inherit Compiler virtually, so there is
// a single Compiler subobject and a GpuLLVMCompiler is usable wherever an
// LLVMCompiler is expected (IR-dump tooling, hook-based tests).
//
// Non-LLVM GPU backends (e.g. Metal) inherit GpuCompiler directly and
// implement its CompileToBackendResult pure virtual without touching
// any of this surface.
class GpuLLVMCompiler : public GpuCompiler, public LLVMCompiler {
 public:
  GpuLLVMCompiler(se::Platform::Id platform_id, const char* target_triple,
                  const char* data_layout);

  // Re-declare names visible through both direct bases (GpuCompiler's
  // overrides; LLVMCompiler's `using Compiler::...` re-exports) so member
  // lookup is unambiguous. Naming the virtual base keeps every Compiler
  // overload visible; virtual dispatch still lands on the GpuCompiler
  // overrides.
  using Compiler::Compile;
  using Compiler::RunBackend;
  using Compiler::RunHloPasses;

  // GpuCompiler::Compile and LLVMCompiler::Compile both override
  // Compiler::Compile (identical bodies: denormal scoping around
  // RunHloPasses + RunBackend); a unique final overrider is required here.
  // Defer to LLVMCompiler's.
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModule> hlo_module,
      std::vector<se::StreamExecutor*> stream_execs,
      const CompileOptions& options) override {
    return LLVMCompiler::Compile(std::move(hlo_module),
                                 std::move(stream_execs), options);
  }

  std::string target_triple() const { return target_triple_; }
  std::string data_layout() const { return data_layout_; }
  const char* GetDataLayout() const { return data_layout_; }
  const char* GetTargetTriple() const { return target_triple_; }

  absl::StatusOr<std::unique_ptr<CompiledModule>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override;

  absl::StatusOr<std::unique_ptr<CompiledModule>> Export(
      Executable* executable) override;

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutableFromAotResult(
      const CompiledModule& aot_result,
      const se::DeviceDescription& device_description) override;

  virtual absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& config,
      const stream_executor::DeviceDescription& device_description,
      se::StreamExecutor* absl_nullable stream_exec) {
    return false;
  }

  // Returns the LLVM command line options that we use for compilation.
  // They need to be set globally whenever we call into LLVM.
  virtual std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const = 0;

  // CUDA-side cub sort/scan scratch-size estimators live here so non-LLVM
  // GPU backends (e.g. Metal) can avoid the cub data link.
  void AddDeviceSpecificScratchSizePasses(
      HloPassPipeline* pipeline,
      const Compiler::GpuTargetConfig& gpu_target_config) override;

 protected:
  struct BackendCompileResult {
    std::string asm_text;
    std::vector<uint8_t> binary;
    BinaryMap dnn_compiled_graphs;
    ModuleStats module_stats;
  };

  // Virtual seam: per-target LLVM-module → backend binary compilation.
  // CUDA / ROCm / SYCL each implement this.
  virtual absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      std::optional<int> shard_number) = 0;

  // Implements GpuCompiler::CompileToBackendResult: produces a fully-built
  // GpuExecutable by running LLVM compilation through CompileSingleModule.
  absl::StatusOr<std::unique_ptr<GpuExecutable>> CompileToBackendResult(
      std::unique_ptr<HloModule> module, const GpuTopology& gpu_topology,
      const CompileOptions& options,
      se::StreamExecutor* absl_nullable stream_exec) override;

  absl::StatusOr<BackendCompileResult> CompileSingleModule(
      const HloModuleConfig& module_config,
      const stream_executor::DeviceDescription& device_description,
      const HloModule* debug_module, llvm::Module* llvm_module,
      bool relocatable, std::optional<int> shard_number);

  // Legacy AOT compilation path. Overrides the non-LLVM stub default that
  // GpuCompiler provides.
  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  LegacyCompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                           const AotCompilationOptions& options) override;

  virtual absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const stream_executor::DeviceDescription& device_description,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options,
      se::StreamExecutor* absl_nullable stream_exec) {
    return Unimplemented("LinkModules is not implemented.");
  }

  ObjectPool<std::unique_ptr<mlir::MLIRContext>>& mlir_context_pool() {
    return mlir_context_pool_;
  }

 private:
  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  GpuLLVMCompiler(const GpuLLVMCompiler&) = delete;
  GpuLLVMCompiler& operator=(const GpuLLVMCompiler&) = delete;

  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_LLVM_COMPILER_H_
