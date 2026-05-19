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
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/metal/codegen/msl_kernel_emitter.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/transforms/passes.h"
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
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace metal {

MetalCompiler::MetalCompiler()
    : xla::gpu::GpuCompiler(stream_executor::metal::kMetalPlatformId,
                            /*pointer_size=*/8) {}

absl::Status MetalCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options,
    const xla::gpu::GpuTargetConfig& gpu_target_config,
    const xla::gpu::GpuAliasInfo* alias_info,
    tsl::thread::ThreadPool* thread_pool, CompilationStats* compilation_stats) {
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
      thread_pool, compilation_stats);
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
  TF_RETURN_IF_ERROR(
      ScheduleAndVerify(hlo_module.get(), gpu_topology, alias_info.get())
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

      case HloOpcode::kFusion: {
        const auto* fusion_instr = Cast<HloFusionInstruction>(instr);
        gpu::HloFusionAnalysis fusion_analysis =
            gpu::HloFusionAnalysis::Create(*fusion_instr, gpu_device_info);
        gpu::HloFusionInfo fusion_info(fusion_analysis, fusion_instr,
                                       buffer_assignment.get(), *call_graph);
        std::unique_ptr<gpu::FusionInterface> emitter =
            gpu::GetFusionEmitter(fusion_info, mlir_context());
        auto* mlir_fusion = dynamic_cast<gpu::MlirKernelFusion*>(emitter.get());
        if (mlir_fusion == nullptr) {
          return Unimplemented(
              "MetalCompiler::CompileToBackendResult: fusion '%s' uses a "
              "non-MLIR emitter (e.g. Triton / CustomFusion); only MLIR-"
              "kernel fusions are supported on Metal.",
              fusion_instr->name());
        }
        TF_ASSIGN_OR_RETURN(
            emitters::KernelArguments kernel_args,
            emitters::KernelArguments::Create(*buffer_assignment,
                                              gpu::GetDefaultBufferAlignment(),
                                              fusion_instr));
        std::string entry_name =
            llvm_ir::SanitizeFunctionName(std::string(fusion_instr->name()));
        TF_ASSIGN_OR_RETURN(MlirKernelSource mlir_source,
                            mlir_fusion->mlir_kernel_emitter()->Emit(
                                mlir_context(), *fusion_instr, entry_name,
                                buffer_assignment.get()));
        // Lower xla_gpu IR down to SCF + arith + tensor + gpu for the MSL
        // translator. We stop before memref/LLVM lowering — Metal keeps
        // tensors as `device T*`-addressed SSA values.
        {
          mlir::PassManager pm(mlir_source.module().getContext());
          gpu::AddLoopTransformationPasses(
              pm, gpu_device_info,
              mlir_fusion->mlir_kernel_emitter()->unroll_factor());
          // The inliner inside AddLoopTransformationPasses leaves large /
          // multiply-called subcomputations as xla.pure_call; rewrite those to
          // func.call, which the MSL translator emits as device functions.
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
          if (mlir::failed(pm.run(mlir_source.module()))) {
            return absl::InternalError(absl::StrCat(
                "MetalCompiler::CompileToBackendResult: MLIR lowering "
                "pipeline failed on fusion '",
                fusion_instr->name(), "'."));
          }
        }
        TF_ASSIGN_OR_RETURN(metal::MslKernelSource msl_source,
                            metal::EmitMslKernel(mlir_source.module()));
        const gpu::LaunchDimensions launch_dims =
            mlir_fusion->launch_dimensions();
        if (!msl_blob.empty()) {
          msl_blob.append("\n");
        }
        msl_blob.append(msl_source.source());
        thunks.push_back(std::make_unique<gpu::KernelThunk>(
            gpu::Thunk::ThunkInfo{}, msl_source.entry_point(),
            std::move(kernel_args), launch_dims,
            /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0,
            se::gpu::TmaMetadata{}));
        break;
      }

      default:
        return Unimplemented(
            "MetalCompiler::CompileToBackendResult: post-scheduling HLO "
            "opcode '%s' is not yet supported on Metal.",
            HloOpcodeString(instr->opcode()));
    }
  }

  TF_ASSIGN_OR_RETURN(auto output_info,
                      gpu::GetOutputInfo(*hlo_module, *buffer_assignment));

  ProgramShape program_shape =
      entry->ComputeProgramShape(/*include_ids=*/false);
  std::string module_name(hlo_module->name());

  gpu::GpuExecutable::Params params;
  params.executable = std::make_unique<gpu::ThunkExecutor>(std::move(thunks));
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
