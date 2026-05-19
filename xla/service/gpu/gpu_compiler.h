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

#ifndef XLA_SERVICE_GPU_GPU_COMPILER_H_
#define XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotune_results.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compilation_stats.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// The GPU compiler generates efficient GPU executables. This is the LLVM-free
// base for all GPU backends: it owns HLO pipeline orchestration and exposes
// a CompileToBackendResult pure-virtual seam that subclasses fill in. LLVM-
// flavored GPU backends (CUDA, ROCm, SYCL/Intel) inherit GpuLLVMCompiler,
// which inherits this and implements the seam via LLVM-module codegen.
// Non-LLVM GPU backends (e.g. Metal) inherit this directly.
class GpuCompiler : public Compiler {
 public:
  using AsmModuleHook = absl::AnyInvocable<void(absl::string_view)>;

  GpuCompiler(se::Platform::Id platform_id, int64_t pointer_size);

  // An attached device is passed in via stream_exec. We get GPU configuration
  // from the attached device OR from the `options` struct (in which case the
  // attached device is ignored during the compilation).
  // If you call this directly, follow it with RunBackend rather than Compile.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  CompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                     AotCompilationOptions const& options) override;

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  absl::Status RunPostSchedulingPipelines(HloModule* module,
                                          int64_t scheduler_mem_limit,
                                          const GpuTopology& gpu_topology,
                                          const GpuAliasInfo* alias_info,
                                          mlir::MLIRContext* mlir_context);

  int64_t GetPointerSize() const { return pointer_size_; }

  virtual std::unique_ptr<GpuAliasInfo> GetAliasInfo(
      const se::DeviceDescription& device_description) const {
    return std::make_unique<GpuAliasInfo>(device_description);
  }

  enum class AlgebraicSimplifierMode {
    kLayoutInsensitive,
    kPostFusionSimplification,
    kLayoutNormalization,
    kPostLayoutAssignment,
    kAfterSimplifyFPConversions,
    kGpuConvoluationCanonicalization,
  };

  static AlgebraicSimplifierOptions GetAlgebraicSimplifierOptions(
      AlgebraicSimplifierMode mode, const DebugOptions& debug_options,
      bool is_rocm);

  // Sets a callback that is invoked with all compiled ptx.
  // Can be used for logging or statistics collection.
  void SetAsmHook(AsmModuleHook hook) {
    absl::MutexLock lock(user_asm_hook_m_);
    user_asm_hook_ = std::move(hook);
  }
  void RemoveAsmHook() {
    absl::MutexLock lock(user_asm_hook_m_);
    user_asm_hook_ = nullptr;
  }

 protected:
  // Virtual seam: produce a fully-built GpuExecutable for a scheduled module.
  // The shared RunBackend handles topology inference, annotations, and post-
  // build dumping; this hook owns codegen and executable construction.
  virtual absl::StatusOr<std::unique_ptr<GpuExecutable>> CompileToBackendResult(
      std::unique_ptr<HloModule> module, const GpuTopology& gpu_topology,
      const CompileOptions& options,
      se::StreamExecutor* absl_nullable stream_exec) = 0;

  // Runs the pre-scheduling passes, the scheduler, the scheduled-module HLO
  // verifier, and the post-scheduling pipelines. Subclasses call this from
  // their CompileToBackendResult override before invoking kernel codegen.
  absl::StatusOr<ScheduleMetadata> ScheduleAndVerify(
      HloModule* module, const GpuTopology& gpu_topology,
      const GpuAliasInfo* alias_info, mlir::MLIRContext* mlir_context);

  void CallUserAsmHook(absl::string_view asm_text) {
    absl::MutexLock lock(user_asm_hook_m_);
    if (user_asm_hook_ && !asm_text.empty()) {
      user_asm_hook_(asm_text);
    }
  }

  static std::unique_ptr<HloPassPipeline> GetCustomKernelRewriterPipeline(
      const stream_executor::DeviceDescription& device_description);

  // Run right before GemmRewriter to add pads for gpublas gemms. Default no-op;
  // LLVM-flavored GPU subclasses (CUDA/ROCm/Intel) override.
  virtual void AddPaddingForGpublasGemms(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version) {}

  // Adds passes that rewrite GEMMs into gpublas custom calls. Non-gpublas
  // backends override this to keep dots in the elemental fusion path.
  virtual void AddGemmRewriteCustomCallPasses(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      se::GpuComputeCapability gpu_version,
      const se::SemanticVersion& toolkit_version);

  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  // thread_pool is used to speed up compilation during autotuning.
  virtual absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
      const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
      CompilationStats* compilation_stats, mlir::MLIRContext* mlir_context);

  virtual absl::Status AddAutotunerPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      stream_executor::StreamExecutor* stream_executor,
      const GpuTargetConfig* target_config, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store);

  // TODO(b/511979384): Remove once xla_gpu_experimental_autotune_post_fusion is
  // enabled by default.
  virtual absl::Status AddConvAndGemmAutotuningPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      se::StreamExecutor* stream_exec,
      const Compiler::GpuTargetConfig* target_config,
      const MultiProcessKeyValueStore& key_value_store,
      const se::SemanticVersion& toolkit_version, const AliasInfo* alias_info,
      const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn);

  // TODO(b/511979384): Remove once xla_gpu_experimental_autotune_post_fusion is
  // enabled by default.
  absl::Status AddFusionAutotuningPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      stream_executor::StreamExecutor* stream_executor,
      const Compiler::GpuTargetConfig* target_config,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store);

  // TODO(b/511979384): Remove once xla_gpu_experimental_autotune_post_fusion is
  // enabled by default.
  absl::Status AutotunerAndPostCleanup(
      HloPassPipeline& pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
      const se::DeviceDescription& device_description,
      const std::string& platform_name, const CompileOptions& options,
      tsl::thread::ThreadPool* thread_pool, se::StreamExecutor* stream_exec,
      const Compiler::GpuTargetConfig* target_config,
      const MultiProcessKeyValueStore& key_value_store,
      const se::SemanticVersion& toolkit_version, const AliasInfo* alias_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn);

  // Runs cuDNN fusion and custom call compiler passes. Default no-op;
  // NVIDIA-side override does the real work.
  virtual absl::Status RunCudnnCompilerPasses(HloModule* module,
                                              se::dnn::DnnSupport& dnn_support,
                                              BinaryMap* dnn_compiled_graphs) {
    return absl::OkStatus();
  }

  // Legacy AOT compilation path. LLVM-flavored subclasses override this; the
  // default returns Unimplemented so non-LLVM GPU backends (e.g. Metal) can
  // skip the legacy path entirely.
  virtual absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  LegacyCompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                           const AotCompilationOptions& options) {
    return Unimplemented("LegacyCompileAheadOfTime is not implemented.");
  }

  // Hook for adding backend-specific scratch-size-estimation passes at the
  // tail of the layout-assignment pipeline. CUDA-side cub sort/scan
  // scratch-size estimators live here so Metal can avoid the cub data link.
  // Default: no-op.
  virtual void AddDeviceSpecificScratchSizePasses(
      HloPassPipeline* pipeline,
      const Compiler::GpuTargetConfig& gpu_target_config) {}

 private:
  absl::Status LoadAutotuneResultsFromFile(const DebugOptions& debug_options);
  absl::Status SerializeAutotuneResultsToFile(
      const DebugOptions& debug_options);

  absl::Status RunPreSchedulingPasses(
      HloModule* module, const se::DeviceDescription& gpu_device_info,
      const GpuAliasInfo* alias_info, mlir::MLIRContext* mlir_context);
  absl::Status RunCollectiveScheduleLinearizerPasses(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      CompilationStats* compilation_stats);

  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  absl::Status OptimizeHloModule(HloModule* hlo_module,
                                 se::StreamExecutor* stream_exec,
                                 const CompileOptions& options,
                                 const GpuTopology& gpu_topology,
                                 const GpuAliasInfo* alias_info,
                                 CompilationStats* compilation_stats);

  // Convolution canonicalization for backends with a cuDNN-equivalent. Default
  // no-op; LLVM-flavored GPU subclasses (CUDA/ROCm/Intel) override.
  virtual absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version,
      CompilationStats* compilation_stats) {
    return absl::OkStatus();
  }

  // Inserts and optimizes mandatory copies. Necessary for correctness.
  absl::Status RunPreSchedulingCopyInsertion(
      HloModule& hlo_module, const se::DeviceDescription& device_description,
      const GpuAliasInfo* alias_info);

  // Runs HLO passes on the given module. If the module has a schedule, it is
  // assumed that the module is already optimized and no passes are run.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPassesIfNeeded(
      std::unique_ptr<HloModule> hlo_module,
      se::StreamExecutor* absl_nullable executor,
      const CompileOptions& compile_options);

  // New AOT compilation which compiles up the the Thunk generation stage.
  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  NewCompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                        se::StreamExecutor* executor,
                        const CompileOptions& compile_options);

  se::Platform::Id platform_id_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;

  // A MLIR context used by pre-codegen passes (constant-fold, lowering
  // helpers). Codegen subclasses use their own contexts.
  mlir::MLIRContext mlir_context_;

  absl::Mutex user_asm_hook_m_;
  AsmModuleHook user_asm_hook_ ABSL_GUARDED_BY(user_asm_hook_m_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_COMPILER_H_
