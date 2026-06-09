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

#include "xla/backends/metal/compiler.h"

#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/metal/codegen/msl_kernel_emitter.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
#include "xla/backends/metal/runtime/metal_kernel_artifact.h"
#include "xla/backends/metal/runtime/metal_kernel_thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/ir_printing.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/name_uniquer.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/metal/metal_device_handle.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/metal/metal_pso_probe.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/maybe_owning.h"
#include "xla/util.h"

namespace xla {
namespace metal {
namespace {

using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

// Mirror of gpu_compiler.cc's factory: explicit --xla_gpu_force_compilation_-
// parallelism wins, otherwise fall back to the caller-supplied pool, otherwise
// spin up one sized to MaxParallelism (or run serial if both are 0/1).
MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  CHECK(default_thread_pool == nullptr ||
        default_thread_pool->CurrentThreadId() == -1);

  auto create_thread_pool = [&](int num_threads) {
    CHECK_GE(num_threads, 1);
    return std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "",
                                                     num_threads);
  };
  switch (parallelism) {
    case 0:
      if (default_thread_pool == nullptr && default_parallelism > 1) {
        return MaybeOwningThreadPool(create_thread_pool(default_parallelism));
      }
      return MaybeOwningThreadPool(default_thread_pool);
    case 1:
      return MaybeOwningThreadPool(nullptr);
    default:
      return MaybeOwningThreadPool(create_thread_pool(parallelism));
  }
}

// Captured by the serial HLO walk; processed in parallel after.
struct DeferredFusion {
  std::string fusion_name;                  // for diagnostics
  const HloFusionInstruction* fusion_instr;  // not owned
  std::string entry_name;                   // globally unique
  emitters::KernelArguments kernel_args;
  xla::metal::MetalKernelThunk* thunk;      // owned by ThunkExecutor
};

}  // namespace

MetalCompiler::MetalCompiler()
    : xla::gpu::GpuCompiler(stream_executor::metal::kMetalPlatformId,
                            /*pointer_size=*/8) {}

absl::Status MetalCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options,
    const xla::gpu::GpuTargetConfig& gpu_target_config,
    const xla::gpu::GpuAliasInfo* alias_info,
    tsl::thread::ThreadPool* thread_pool, CompilationStats* compilation_stats,
    mlir::MLIRContext* mlir_context) {
  // Widen low-precision types the MSL emitter cannot lower. The base
  // GpuCompiler's float_normalization sub-pipeline keys its decisions off
  // CUDA/ROCm compute capabilities and would otherwise leave many of these
  // types (BF16 data movement, F8 data movement, BF16 dot) in HLO when run
  // against a MetalComputeCapability. Mirrors AMDGPUCompiler's pattern of
  // running a target-specific FloatNormalization pre-pipeline before
  // delegating to the base.
  HloPassPipeline pre("metal_pre_normalization", compilation_stats);
  FloatSupport bf16(BF16);
  FloatSupport f8e5m2(F8E5M2, F16);
  FloatSupport f8e4m3(F8E4M3, F16);
  FloatSupport f8e3m4(F8E3M4, F16);
  FloatSupport f8e4m3fn(F8E4M3FN, F16);
  FloatSupport f8e4m3fnuz(F8E4M3FNUZ, F16);
  FloatSupport f8e5m2fnuz(F8E5M2FNUZ, F16);
  FloatSupport f8e4m3b11fnuz(F8E4M3B11FNUZ, F16);
  FloatSupport f4e2m1fn(F4E2M1FN, F16);
  FloatSupport f8e8m0fnu(F8E8M0FNU, F16);
  pre.AddPass<FloatNormalization>(&bf16);
  pre.AddPass<FloatNormalization>(&f8e5m2);
  pre.AddPass<FloatNormalization>(&f8e4m3);
  pre.AddPass<FloatNormalization>(&f8e3m4);
  pre.AddPass<FloatNormalization>(&f8e4m3fn);
  pre.AddPass<FloatNormalization>(&f8e4m3fnuz);
  pre.AddPass<FloatNormalization>(&f8e5m2fnuz);
  pre.AddPass<FloatNormalization>(&f8e4m3b11fnuz);
  pre.AddPass<FloatNormalization>(&f4e2m1fn);
  pre.AddPass<FloatNormalization>(&f8e8m0fnu);
  TF_RETURN_IF_ERROR(
      pre.Run(hlo_module,
              /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  return xla::gpu::GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, alias_info,
      thread_pool, compilation_stats, mlir_context);
}

void MetalCompiler::AddGemmRewriteCustomCallPasses(
    HloPassPipeline&, const DebugOptions&, se::GpuComputeCapability,
    const se::SemanticVersion&) {
  // Metal has no gpublas custom-call implementation. Keep dots in HLO so they
  // can be wrapped into MLIR elemental fusions instead.
}

absl::StatusOr<std::unique_ptr<xla::gpu::GpuExecutable>>
MetalCompiler::CompileToBackendResult(std::unique_ptr<HloModule> hlo_module,
                                      const GpuTopology& gpu_topology,
                                      const CompileOptions& /*options*/,
                                      se::StreamExecutor* stream_exec) {
  VLOG(1) << "MetalCompiler::CompileToBackendResult on " << hlo_module->name();
  if (stream_exec == nullptr) {
    return absl::InvalidArgumentError(
        "MetalCompiler::CompileToBackendResult requires a non-null "
        "StreamExecutor.");
  }
  HloComputation* entry = hlo_module->entry_computation();
  if (entry == nullptr) {
    return absl::InvalidArgumentError(
        "MetalCompiler::CompileToBackendResult: HloModule has no entry "
        "computation.");
  }

  const se::DeviceDescription& gpu_device_info =
      gpu_topology.gpu_target_config().device_description;
  std::unique_ptr<gpu::GpuAliasInfo> alias_info = GetAliasInfo(gpu_device_info);

  // Schedule via the inherited GpuCompiler helper (pre-scheduling passes,
  // scheduler, scheduled-module verifier, post-scheduling pipelines), then
  // use SequentialHloOrdering off the resulting schedule for buffer
  // assignment — same shape as the LLVM-flavored GPU path.
  TF_RETURN_IF_ERROR(ScheduleAndVerify(hlo_module.get(), gpu_topology,
                                       alias_info.get(), mlir_context())
                         .status());

  BufferAssigner::Options buffer_assigner_options;
  // Match compile_module_to_llvm_ir.cc:180 — without this, BufferAssigner
  // skips slice allocation for kConstant HLOs, and our kConstant handler's
  // GetUniqueSlice lookup fails with "Slice not assigned".
  buffer_assigner_options.allocate_buffers_for_constants = true;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          hlo_module.get(),
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          BufferSizeBytesFunction(), alias_info.get(),
          /*color_alignment=*/
          [](LogicalBuffer::Color) {
            return gpu::kXlaAllocatedBufferAlignBytes;
          },
          std::move(buffer_assigner_options)));

  // Per-opcode thunk emission mirrors xla::gpu::ThunkEmitter. For kFusion:
  // MlirKernelEmitter → metal::EmitMslKernel → KernelThunk; per-fusion MSL
  // accumulates into GpuExecutable::Params::asm_text (NVPTX uses the same
  // slot for PTX). Dialect registry mirrors MlirKernelFusion's needs.
  mlir_context()->appendDialectRegistry(
      gpu::MlirKernelEmitter::GetDialectRegistry());
  mlir_context()->loadAllAvailableDialects();
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module.get());
  gpu::ThunkSequence thunks;
  std::vector<gpu::GpuExecutable::ConstantInfo> constants;
  std::string msl_blob;
  NameUniquer msl_function_name_uniquer;
  std::vector<DeferredFusion> deferred_fusions;
  for (const HloInstruction* instr :
       hlo_module->schedule().sequence(entry).instructions()) {
    switch (instr->opcode()) {
      // Encoded by buffer assignment / ordering; no thunk needed.
      case HloOpcode::kAddDependency:
      case HloOpcode::kAfterAll:
      case HloOpcode::kBitcast:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kParameter:
      case HloOpcode::kTuple:
        break;

      case HloOpcode::kConstant: {
        // Mirrors ThunkEmitter::EmitConstant: no thunk; the literal bytes
        // land in params.constants, and GpuExecutable copies them into the
        // allocation at initialize time.
        const auto* constant_instr = Cast<HloConstantInstruction>(instr);
        TF_ASSIGN_OR_RETURN(gpu::DenseDataIntermediate content,
                            gpu::LiteralToXlaFormat(constant_instr->literal()));
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice slice,
            buffer_assignment->GetUniqueSlice(constant_instr, /*index=*/{}));
        gpu::GpuExecutable::ConstantInfo info;
        info.symbol_name = llvm_ir::ConstantHloToGlobalName(*constant_instr);
        info.content = std::move(content);
        info.allocation_index = slice.index();
        constants.push_back(std::move(info));
        break;
      }

      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice src,
            buffer_assignment->GetUniqueSlice(instr->operand(0), /*index=*/{}));
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice dst,
            buffer_assignment->GetUniqueSlice(instr, /*index=*/{}));
        thunks.push_back(std::make_unique<gpu::DeviceToDeviceCopyThunk>(
            gpu::Thunk::ThunkInfo{},
            /*source_buffer=*/ShapedSlice{src, instr->operand(0)->shape()},
            /*destination_buffer=*/ShapedSlice{dst, instr->shape()},
            /*mem_size=*/src.size()));
        break;
      }

      case HloOpcode::kFft: {
        // The MPSGraph-backed MetalFft plugin is registered against
        // kMetalPlatformId, so gpu::FftThunk's runtime resolves AsFft()
        // and executes through the shared runtime path used by CUDA.
        const auto* fft = Cast<HloFftInstruction>(instr);
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice arg_slice,
            buffer_assignment->GetUniqueSlice(fft->operand(0), {}));
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                            buffer_assignment->GetUniqueSlice(fft, {}));
        thunks.push_back(std::make_unique<gpu::FftThunk>(
            gpu::Thunk::ThunkInfo{}, fft->fft_type(), fft->fft_length(),
            /*input_buffer=*/arg_slice,
            /*output_buffer=*/dest_slice,
            /*input_shape=*/fft->operand(0)->shape(),
            /*output_shape=*/fft->shape()));
        break;
      }

      case HloOpcode::kFusion: {
        const auto* fusion_instr = Cast<HloFusionInstruction>(instr);
        TF_ASSIGN_OR_RETURN(
            emitters::KernelArguments kernel_args,
            emitters::KernelArguments::Create(*buffer_assignment,
                                              gpu::GetDefaultBufferAlignment(),
                                              fusion_instr));
        // Reserve a globally-unique entry name now so Phase 2 can build the
        // PSO against this name. Helper-function names get prefixed with the
        // entry name in EmitMslKernel, so a globally-unique entry plus a
        // per-fusion uniquer keeps helpers collision-free across fusions
        // without a shared lock.
        std::string entry_name = msl_function_name_uniquer.GetUniqueName(
            llvm_ir::SanitizeFunctionName(std::string(fusion_instr->name())));
        auto thunk = std::make_unique<xla::metal::MetalKernelThunk>(
            gpu::Thunk::ThunkInfo{}, kernel_args);
        xla::metal::MetalKernelThunk* thunk_ptr = thunk.get();
        thunks.push_back(std::move(thunk));
        deferred_fusions.push_back(DeferredFusion{
            std::string(fusion_instr->name()), fusion_instr,
            std::move(entry_name), std::move(kernel_args), thunk_ptr});
        break;
      }

      default:
        return Unimplemented(
            "MetalCompiler::CompileToBackendResult: post-scheduling HLO "
            "opcode '%s' is not yet supported on Metal.",
            HloOpcodeString(instr->opcode()));
    }
  }

  // Phase 2: per-fusion retry loop. Each worker emits MLIR with a derived
  // device description that caps threads_per_block_limit at the current hint,
  // lowers, translates to MSL, then compiles + probes the PSO. If the PSO
  // grants fewer threads than the launch requests, we drop the hint to a
  // multiple of SIMD width at-or-below the PSO ceiling and retry. Floors at
  // one SIMD group; if even that won't host the kernel, surface a clean
  // ResourceExhaustedError naming the fusion. An emitter that ignores the
  // soft cap (num_threads_per_block stays put across retries) trips the
  // unchanged-requested-threads guard so we bail instead of spinning.
  void* metal_device =
      stream_executor::metal::GetMetalDeviceOpaque(stream_exec);
  if (metal_device == nullptr) {
    return absl::FailedPreconditionError(
        "MetalCompiler::CompileToBackendResult: stream executor is not a "
        "MetalExecutor or device handle is unavailable.");
  }

  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      hlo_module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/nullptr,
      /*default_parallelism=*/static_cast<int>(
          std::thread::hardware_concurrency() > 0
              ? std::thread::hardware_concurrency()
              : 1));

  struct BuildResult {
    std::unique_ptr<xla::metal::MetalKernelArtifact> artifact;
    std::string msl;
  };
  auto build_artifact = [&](const DeferredFusion& deferred,
                            int64_t initial_max_threads_per_block)
      -> absl::StatusOr<BuildResult> {
    const int64_t simd_width = gpu_device_info.threads_per_warp();
    int64_t hint = initial_max_threads_per_block;
    int64_t last_requested = -1;
    std::string last_msl;
    for (;;) {
      if (hint < simd_width) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Fusion '", deferred.fusion_name,
            "': cannot satisfy PSO maxTotalThreadsPerThreadgroup at any "
            "threadgroup size >= ",
            simd_width,
            " (SIMD width). Per-kernel register pressure too high on this "
            "device."));
      }
      se::DeviceDescription dev = gpu_device_info;
      dev.set_threads_per_block_limit(hint);

      auto context = std::make_unique<mlir::MLIRContext>();
      context->appendDialectRegistry(
          gpu::MlirKernelEmitter::GetDialectRegistry());
      context->loadAllAvailableDialects();

      gpu::HloFusionAnalysis fusion_analysis =
          gpu::HloFusionAnalysis::Create(*deferred.fusion_instr, dev);
      gpu::HloFusionInfo fusion_info(fusion_analysis, deferred.fusion_instr,
                                     buffer_assignment.get(), *call_graph);
      std::unique_ptr<gpu::FusionInterface> emitter =
          gpu::GetFusionEmitter(fusion_info, context.get());
      auto* mlir_fusion = dynamic_cast<gpu::MlirKernelFusion*>(emitter.get());
      if (mlir_fusion == nullptr) {
        return Unimplemented(
            "MetalCompiler::CompileToBackendResult: fusion '%s' uses a "
            "non-MLIR emitter (e.g. Triton / CustomFusion); only MLIR-"
            "kernel fusions are supported on Metal.",
            deferred.fusion_name);
      }
      TF_ASSIGN_OR_RETURN(MlirKernelSource mlir_source,
                          mlir_fusion->mlir_kernel_emitter()->Emit(
                              context.get(), *deferred.fusion_instr,
                              deferred.entry_name, buffer_assignment.get()));
      gpu::LaunchDimensions launch_dims = mlir_fusion->launch_dimensions();
      int unroll_factor =
          mlir_fusion->mlir_kernel_emitter()->unroll_factor();

      mlir::ModuleOp module = mlir_source.module();
      mlir::PassManager pm(module.getContext());
      gpu::AddLoopTransformationPasses(pm, dev, unroll_factor,
                                       /*max_vector_elements=*/4);
      pm.addNestedPass<mlir::func::FuncOp>(
          emitters::CreateConvertPureCallOpsPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          emitters::CreateSimplifyArithPass());
      pm.addPass(emitters::CreateSimplifyAffinePass());
      pm.addPass(gpu::CreateConvertIndexTypePass());
      pm.addPass(mlir::createLowerAffinePass());
      pm.addPass(mlir::createLoopInvariantCodeMotionPass());
      pm.addPass(mlir::createSymbolDCEPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(CreateConvertComplexToArithMathPass());
      pm.addPass(emitters::CreateExpandFloatOpsPass());
      pm.addPass(CreateExpandFloatOpsPass());
      pm.addPass(CreateLowerSubByteStoragePass());
      pm.addPass(CreateLowerFloatStoragePass());
      pm.addPass(mlir::createLowerAffinePass());
      std::string dump_kernel_name =
          absl::StrCat(deferred.entry_name, ".metal-lowering");
      EnableIRPrintingIfRequested(pm, module.getContext(), *hlo_module,
                                  dump_kernel_name, "mlir-fusion");
      if (mlir::failed(pm.run(module))) {
        return absl::InternalError(absl::StrCat(
            "MetalCompiler::CompileToBackendResult: MLIR lowering pipeline "
            "failed on fusion '",
            deferred.fusion_name, "'."));
      }
      NameUniquer per_fusion_uniquer;
      TF_ASSIGN_OR_RETURN(metal::MslKernelSource msl_source,
                          metal::EmitMslKernel(module, &per_fusion_uniquer,
                                               *hlo_module,
                                               deferred.entry_name));
      last_msl = std::move(msl_source).source();

      TF_ASSIGN_OR_RETURN(
          stream_executor::metal::CompiledPipeline compiled,
          stream_executor::metal::CompileAndProbe(metal_device, last_msl,
                                                  deferred.entry_name, hint));
      int64_t requested = launch_dims.num_threads_per_block();
      if (last_requested == requested && requested > compiled.pso_max_threads) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Fusion '", deferred.fusion_name, "': emitter requires ",
            requested,
            " threads per threadgroup, but the device's PSO grants only ",
            compiled.pso_max_threads,
            " (register pressure). The emitter for this fusion does not "
            "honor the threads_per_block_limit soft cap, so retry cannot "
            "shrink the threadgroup."));
      }
      last_requested = requested;
      if (compiled.pso_max_threads >= requested) {
        const unsigned arity =
            static_cast<unsigned>(deferred.kernel_args.args().size());
        auto artifact = std::make_unique<xla::metal::MetalKernelArtifact>(
            deferred.entry_name, arity, launch_dims,
            std::move(compiled.pso));
        return BuildResult{std::move(artifact), std::move(last_msl)};
      }
      int64_t next = (compiled.pso_max_threads / simd_width) * simd_width;
      if (next >= hint) {
        next = hint - simd_width;
      }
      VLOG(1) << "Fusion '" << deferred.fusion_name << "': requested "
              << requested << " threads, PSO ceiling "
              << compiled.pso_max_threads << "; retrying with hint " << next;
      hint = next;
    }
  };

  std::vector<absl::StatusOr<BuildResult>> results(deferred_fusions.size());
  if (deferred_fusions.empty()) {
    // Nothing to do.
  } else if (thread_pool.get() == nullptr) {
    for (size_t i = 0; i < deferred_fusions.size(); ++i) {
      results[i] = build_artifact(deferred_fusions[i],
                                  gpu_device_info.threads_per_block_limit());
    }
  } else {
    absl::BlockingCounter counter(deferred_fusions.size());
    for (size_t i = 0; i < deferred_fusions.size(); ++i) {
      thread_pool.get_mutable()->Schedule([&, i] {
        results[i] = build_artifact(deferred_fusions[i],
                                    gpu_device_info.threads_per_block_limit());
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
  for (size_t i = 0; i < deferred_fusions.size(); ++i) {
    TF_RETURN_IF_ERROR(results[i].status());
  }

  // Phase 3: install artifacts on their MetalKernelThunks and concatenate
  // per-fusion MSL into the shared `asm_text` blob for dump consumers. The
  // runtime no longer reads the MSL — MetalKernelThunk dispatches via the
  // artifact's precompiled PSO.
  for (size_t i = 0; i < deferred_fusions.size(); ++i) {
    BuildResult build_result = *std::move(results[i]);
    deferred_fusions[i].thunk->SetArtifact(std::move(build_result.artifact));
    if (!msl_blob.empty()) msl_blob.append("\n");
    msl_blob.append(build_result.msl);
  }

  TF_ASSIGN_OR_RETURN(auto output_info,
                      gpu::GetOutputInfo(*hlo_module, *buffer_assignment));

  ProgramShape program_shape =
      entry->ComputeProgramShape(/*include_ids=*/false);
  std::string module_name(hlo_module->name());

  gpu::GpuExecutable::Params params;
  params.executable = std::make_unique<gpu::ThunkExecutor>(std::move(thunks));
  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "", "msl", msl_blob);
  }
  params.asm_text = std::move(msl_blob);
  params.constants = std::move(constants);
  params.buffer_assignment = std::move(buffer_assignment);
  params.alias_info = std::move(alias_info);
  params.device_description = gpu_device_info;
  params.module_name = std::move(module_name);
  params.program_shape = std::move(program_shape);
  params.output_info = std::move(output_info);
  params.enable_debug_info_manager = false;
  params.debug_module = std::move(hlo_module);

  return gpu::GpuExecutable::Create(std::move(params));
}

static bool InitMetalCompilerModule() {
  Compiler::RegisterCompilerFactory(
      stream_executor::metal::kMetalPlatformId,
      []() { return std::make_unique<MetalCompiler>(); });
  ComputationPlacer::RegisterComputationPlacer(
      stream_executor::metal::kMetalPlatformId,
      []() { return std::make_unique<ComputationPlacer>(); });
  return true;
}

static bool metal_compiler_module_initialized = InitMetalCompilerModule();

}  // namespace metal
}  // namespace xla
