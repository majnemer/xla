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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Intermediate GPU-compiler base class for backends whose codegen path
// terminates in an llvm::Module compiled to a platform-specific binary
// (CUDA → PTX/cubin, ROCm → ELF, Intel/SPIR-V). Sits between GpuCompiler
// (HLO pipeline + GpuExecutable orchestration; backend-agnostic) and the
// concrete LLVM-flavored subclasses (NVPTXCompiler, AMDGPUCompiler,
// IntelGpuCompiler).
//
// Does not inherit LLVMCompiler. Reimplements the small hook-plumbing API
// locally to avoid the GpuCompiler/LLVMCompiler diamond; GPU-side
// `down_cast<LLVMCompiler*>` callers (gpu_opt, gpu_pjrt_codegen_test) cast
// to GpuLLVMCompiler* instead.
class GpuLLVMCompiler : public GpuCompiler {
 public:
  // Callback invoked before and/or after IR-level optimization to dump IR
  // or gather statistics.
  using ModuleHook = std::function<void(const llvm::Module&)>;

  GpuLLVMCompiler(se::Platform::Id platform_id, const char* target_triple,
                  const char* data_layout);

  absl::StatusOr<std::unique_ptr<CompiledModule>> Export(
      Executable* executable) override;

  absl::StatusOr<std::unique_ptr<CompiledModule>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override;

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutableFromAotResult(
      const CompiledModule& aot_result,
      const se::DeviceDescription& device_description) override;

  std::string target_triple() const { return target_triple_; }
  std::string data_layout() const { return data_layout_; }
  const char* GetDataLayout() const { return data_layout_; }
  const char* GetTargetTriple() const { return target_triple_; }

  virtual absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& config,
      const stream_executor::DeviceDescription& device_description,
      se::StreamExecutor* absl_nullable stream_exec) {
    return false;
  }

  // LLVM command line options set globally whenever we call into LLVM.
  virtual std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const = 0;

  void SetPreOptimizationHook(ModuleHook hook) {
    absl::MutexLock lock(hooks_m_);
    CHECK(!user_pre_optimization_hook_)
        << "Pre-optimization hook is already set";
    CHECK(hook) << "hook cannot be null";
    user_pre_optimization_hook_ = hook;
  }

  void RemovePreOptimizationHook() {
    absl::MutexLock lock(hooks_m_);
    user_pre_optimization_hook_ = nullptr;
  }

  void SetPostOptimizationHook(ModuleHook hook) {
    absl::MutexLock lock(hooks_m_);
    CHECK(!user_post_optimization_hook_)
        << "Post-optimization hook is already set";
    CHECK(hook) << "hook cannot be null";
    user_post_optimization_hook_ = hook;
  }

  void RemovePostOptimizationHook() {
    absl::MutexLock lock(hooks_m_);
    user_post_optimization_hook_ = nullptr;
  }

 protected:
  struct BackendCompileResult {
    std::string asm_text;
    std::vector<uint8_t> binary;
    BinaryMap dnn_compiled_graphs;
    ModuleStats module_stats;
  };

  // Virtual seam: LLVM-IR-emit + lower + binary-compile + assemble executable
  // for one scheduled HloModule.
  absl::StatusOr<std::unique_ptr<GpuExecutable>> CompileToBackendResult(
      std::unique_ptr<HloModule> module, const GpuTopology& gpu_topology,
      const CompileOptions& options,
      se::StreamExecutor* absl_nullable stream_exec) override;

  // Runs cuDNN fusion and custom call compiler passes. Default no-op;
  // NVPTXCompiler overrides.
  virtual absl::Status RunCudnnCompilerPasses(HloModule* module,
                                              se::dnn::DnnSupport& dnn_support,
                                              BinaryMap* dnn_compiled_graphs) {
    return absl::OkStatus();
  }

  // Adds the cub sort/scan scratch-size estimation passes (NVIDIA / hipcub).
  // The HLO pipeline keeps these regardless of platform name; the passes
  // themselves no-op at runtime when no cub-targeted FFI handler is
  // registered.
  void AddDeviceSpecificScratchSizePasses(
      HloPassPipeline* pipeline,
      const Compiler::GpuTargetConfig& gpu_target_config) override;

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      std::optional<int> shard_number) = 0;

  virtual absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const stream_executor::DeviceDescription& device_description,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options,
      se::StreamExecutor* absl_nullable stream_exec) {
    return Unimplemented("LinkModules is not implemented.");
  }

  void CallUserPreOptimizationHook(const llvm::Module& module) {
    absl::MutexLock lock(hooks_m_);
    if (user_pre_optimization_hook_) {
      user_pre_optimization_hook_(module);
    }
  }

  void CallUserPostOptimizationHook(const llvm::Module& module) {
    absl::MutexLock lock(hooks_m_);
    if (user_post_optimization_hook_) {
      user_post_optimization_hook_(module);
    }
  }

 private:
  struct CompileResultWithMetadata {
    BackendCompileResult backend_result;
    CompileModuleResults compile_module_results;
  };

  absl::StatusOr<BackendCompileResult> CompileAndLink(
      const HloModuleConfig& module_config,
      CompileModuleResults& compile_module_results,
      const stream_executor::DeviceDescription& device_description,
      const CompileOptions& options, const HloModule* debug_module,
      se::StreamExecutor* absl_nullable stream_exec);

  absl::StatusOr<BackendCompileResult> CompileSingleModule(
      const HloModuleConfig& module_config,
      const stream_executor::DeviceDescription& device_description,
      const HloModule* debug_module, llvm::Module* llvm_module,
      bool relocatable, std::optional<int> shard_number);

  // Drives MLIR-IR emission + LLVM lowering + binary compile for one module.
  // The shared GpuCompiler::RunBackend has already scheduled the module via
  // ScheduleAndVerify before invoking the virtual.
  absl::StatusOr<CompileResultWithMetadata> CompileToBackendResultImpl(
      HloModule* module, llvm::LLVMContext* llvm_context,
      const GpuTopology& gpu_topology, const CompileOptions& options,
      se::StreamExecutor* absl_nullable stream_exec);

  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  LegacyCompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                           const AotCompilationOptions& options) override;

  const char* target_triple_;
  const char* data_layout_;

  absl::Mutex hooks_m_;
  ModuleHook user_pre_optimization_hook_ ABSL_GUARDED_BY(hooks_m_);
  ModuleHook user_post_optimization_hook_ ABSL_GUARDED_BY(hooks_m_);

  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool_;

  GpuLLVMCompiler(const GpuLLVMCompiler&) = delete;
  GpuLLVMCompiler& operator=(const GpuLLVMCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_LLVM_COMPILER_H_
