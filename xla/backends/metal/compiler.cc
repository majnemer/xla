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

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/metal/codegen/msl_kernel_emitter.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/transforms/passes.h"
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
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/thunk_emitter.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace metal {
namespace {

class RngGetAndUpdateStateThunk final : public gpu::Thunk {
 public:
  RngGetAndUpdateStateThunk(ThunkInfo thunk_info, std::string kernel_name,
                            std::string state_symbol_name,
                            BufferAllocation::Slice output_buffer)
      : Thunk(Kind::kCommand, std::move(thunk_info)),
        kernel_name_(std::move(kernel_name)),
        state_symbol_name_(std::move(state_symbol_name)),
        output_buffer_(output_buffer) {}

  absl::Status Initialize(const InitializeParams& params) override {
    absl::MutexLock lock(&mu_);
    if (kernel_cache_.contains(params.executor)) {
      return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        gpu::CreateKernel(kernel_name_, /*num_args=*/2, params.src.text,
                          params.executor));
    if (params.globals == nullptr) {
      return absl::InternalError(
          "Metal RNG state globals were not provided during thunk "
          "initialization");
    }
    auto state_it = params.globals->find(state_symbol_name_);
    if (state_it == params.globals->end()) {
      return absl::NotFoundError(
          absl::StrCat("Metal RNG state global not found: ",
                       state_symbol_name_));
    }
    se::DeviceAddressBase state_data = state_it->second;
    if (state_data.size() != 2 * sizeof(uint64_t)) {
      return InvalidArgument("Invalid Metal RNG state symbol size: %d",
                             state_data.size());
    }

    kernel_cache_.emplace(params.executor,
                          KernelAndState{std::move(kernel), state_data});
    return absl::OkStatus();
  }

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    se::Kernel* kernel;
    se::DeviceAddressBase state_data;
    {
      absl::MutexLock lock(&mu_);
      auto it = kernel_cache_.find(params.stream->parent());
      if (it == kernel_cache_.end() || it->second.kernel == nullptr) {
        return absl::InternalError(absl::StrCat(
            "Metal RNG kernel not loaded for executor: ", kernel_name_));
      }
      kernel = it->second.kernel.get();
      state_data = it->second.state_data;
    }

    se::DeviceMemoryBase output_data =
        params.buffer_allocations->GetDeviceAddress(output_buffer_);
    if (output_data.size() != 2 * sizeof(uint64_t)) {
      return InvalidArgument("Invalid RNG output buffer size: %d",
                             output_data.size());
    }

    std::vector<se::KernelArg> args = {state_data, output_data};
    return gpu::ExecuteKernelOnStream(
        *kernel, args, gpu::LaunchDimensions(), /*cluster_dim=*/std::nullopt,
        params.stream);
  }

  BufferUses buffer_uses() const override {
    return {BufferUse::Write(output_buffer_, ShapeUtil::MakeShape(U64, {2}))};
  }

 private:
  struct KernelAndState {
    std::unique_ptr<se::Kernel> kernel;
    se::DeviceAddressBase state_data;
  };

  std::string kernel_name_;
  std::string state_symbol_name_;
  BufferAllocation::Slice output_buffer_;

  absl::Mutex mu_;
  absl::flat_hash_map<se::StreamExecutor*, KernelAndState> kernel_cache_
      ABSL_GUARDED_BY(mu_);
};

class MetalThunkEmissionBackend final : public gpu::ThunkEmissionBackend {
 public:
  MetalThunkEmissionBackend(HloModule* hlo_module,
                            const se::DeviceDescription& gpu_device_info,
                            BufferAssignment* buffer_assignment,
                            mlir::MLIRContext* mlir_context,
                            CallGraph* call_graph,
                            std::vector<gpu::GpuExecutable::ConstantInfo>*
                                constants,
                            std::vector<gpu::GpuExecutable::GlobalInfo>*
                                globals,
                            std::string* msl_blob,
                            NameUniquer* msl_function_name_uniquer)
      : hlo_module_(hlo_module),
        gpu_device_info_(gpu_device_info),
        buffer_assignment_(buffer_assignment),
        mlir_context_(mlir_context),
        call_graph_(call_graph),
        constants_(constants),
        globals_(globals),
        msl_blob_(msl_blob),
        msl_function_name_uniquer_(msl_function_name_uniquer) {}

  const BufferAssignment& buffer_assignment() const override {
    return *buffer_assignment_;
  }

  gpu::Thunk::ThunkInfo GetThunkInfo(const HloInstruction* instr) override {
    return gpu::Thunk::ThunkInfo::WithProfileAnnotation(
        instr, thunk_id_generator_.GetNextThunkId());
  }

  gpu::AsyncThunkSequence EmitTargetElement(
      const HloInstruction* instr, bool emit_group_thunks) override {
    (void)emit_group_thunks;
    switch (instr->opcode()) {
      case HloOpcode::kConstant:
        return EmitConstant(Cast<HloConstantInstruction>(instr));
      case HloOpcode::kFusion:
        return EmitFusion(Cast<HloFusionInstruction>(instr));
      case HloOpcode::kRngGetAndUpdateState:
        return EmitRngGetAndUpdateState(
            Cast<HloRngGetAndUpdateStateInstruction>(instr));
      default:
        return Unimplemented(
            "MetalCompiler::CompileToBackendResult: post-scheduling HLO "
            "opcode '%s' is not yet supported on Metal.",
            HloOpcodeString(instr->opcode()));
    }
  }

 private:
  absl::StatusOr<gpu::ThunkSequence> EmitConstant(
      const HloConstantInstruction* constant_instr) {
    if (!emitted_constants_.insert(constant_instr).second) {
      return gpu::ThunkSequence{};
    }
    TF_ASSIGN_OR_RETURN(gpu::DenseDataIntermediate content,
                        gpu::LiteralToXlaFormat(constant_instr->literal()));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment_->GetUniqueSlice(constant_instr,
                                                           /*index=*/{}));
    gpu::GpuExecutable::ConstantInfo info;
    info.symbol_name = llvm_ir::ConstantHloToGlobalName(*constant_instr);
    info.content = std::move(content);
    info.allocation_index = slice.index();
    constants_->push_back(std::move(info));
    return gpu::ThunkSequence{};
  }

  absl::StatusOr<gpu::ThunkSequence> EmitFusion(
      const HloFusionInstruction* fusion_instr) {
    gpu::HloFusionAnalysis fusion_analysis =
        gpu::HloFusionAnalysis::Create(*fusion_instr, gpu_device_info_);
    gpu::HloFusionInfo fusion_info(fusion_analysis, fusion_instr,
                                   buffer_assignment_, *call_graph_);
    std::unique_ptr<gpu::FusionInterface> emitter =
        gpu::GetFusionEmitter(fusion_info, mlir_context_);
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
        emitters::KernelArguments::Create(*buffer_assignment_,
                                          gpu::GetDefaultBufferAlignment(),
                                          fusion_instr));
    std::string entry_name =
        llvm_ir::SanitizeFunctionName(std::string(fusion_instr->name()));
    TF_ASSIGN_OR_RETURN(MlirKernelSource mlir_source,
                        mlir_fusion->mlir_kernel_emitter()->Emit(
                            mlir_context_, *fusion_instr, entry_name,
                            buffer_assignment_));
    // Lower xla_gpu IR down to SCF + arith + tensor + gpu for the MSL
    // translator. We stop before memref/LLVM lowering — Metal keeps tensors as
    // `device T*`-addressed SSA values.
    {
      mlir::PassManager pm(mlir_source.module().getContext());
      gpu::AddLoopTransformationPasses(
          pm, gpu_device_info_,
          mlir_fusion->mlir_kernel_emitter()->unroll_factor(),
          /*max_vector_elements=*/4);
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
      std::string dump_kernel_name =
          absl::StrCat(entry_name, ".metal-lowering");
      EnableIRPrintingIfRequested(pm, mlir_source.module().getContext(),
                                  *hlo_module_, dump_kernel_name,
                                  "mlir-fusion");
      if (mlir::failed(pm.run(mlir_source.module()))) {
        return absl::InternalError(absl::StrCat(
            "MetalCompiler::CompileToBackendResult: MLIR lowering "
            "pipeline failed on fusion '",
            fusion_instr->name(), "'."));
      }
    }
    TF_ASSIGN_OR_RETURN(
        metal::MslKernelSource msl_source,
        metal::EmitMslKernel(mlir_source.module(), msl_function_name_uniquer_,
                             *hlo_module_, entry_name));
    const gpu::LaunchDimensions launch_dims = mlir_fusion->launch_dimensions();
    if (!msl_blob_->empty()) {
      msl_blob_->append("\n");
    }
    msl_blob_->append(msl_source.source());

    gpu::ThunkSequence thunks;
    thunks.push_back(std::make_unique<gpu::KernelThunk>(
        GetThunkInfo(fusion_instr), msl_source.entry_point(),
        std::move(kernel_args), launch_dims,
        /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0,
        se::gpu::TmaMetadata{}));
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* rng_state) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice output_buffer,
        buffer_assignment_->GetUniqueSlice(rng_state, /*index=*/{}));
    std::string state_symbol_name = GetOrCreateRngStateSymbolName();
    std::string kernel_name = msl_function_name_uniquer_->GetUniqueName(
        llvm_ir::SanitizeFunctionName(std::string(rng_state->name())));
    if (!msl_blob_->empty()) {
      msl_blob_->append("\n");
    }
    msl_blob_->append(EmitRngGetAndUpdateStateMsl(kernel_name,
                                                  rng_state->delta()));

    gpu::ThunkSequence thunks;
    thunks.push_back(std::make_unique<RngGetAndUpdateStateThunk>(
        GetThunkInfo(rng_state), std::move(kernel_name),
        std::move(state_symbol_name), output_buffer));
    return thunks;
  }

  std::string GetOrCreateRngStateSymbolName() {
    if (rng_state_symbol_name_.has_value()) {
      return *rng_state_symbol_name_;
    }
    rng_state_symbol_name_ = llvm_ir::SanitizeFunctionName(absl::StrCat(
        hlo_module_->name(), "_", hlo_module_->unique_id(), "_rng_state"));

    std::array<uint64_t, 2> initial_state = {0x7012395ull, 0};
    std::vector<uint8_t> content(sizeof(initial_state));
    std::memcpy(content.data(), initial_state.data(), content.size());

    gpu::GpuExecutable::GlobalInfo info;
    info.symbol_name = *rng_state_symbol_name_;
    info.initial_value = gpu::DenseDataIntermediate::Own(std::move(content));
    globals_->push_back(std::move(info));
    return *rng_state_symbol_name_;
  }

  static std::string EmitRngGetAndUpdateStateMsl(absl::string_view kernel_name,
                                                 int64_t delta) {
    return absl::Substitute(R"msl(#include <metal_stdlib>
using namespace metal;

kernel void $0(device ulong* rng_state [[buffer(0)]],
                       device ulong* output [[buffer(1)]]) {
  ulong old_low = rng_state[0];
  ulong old_high = rng_state[1];
  output[0] = old_low;
  output[1] = old_high;

  ulong new_low = old_low + $1ul;
  ulong carry = new_low < old_low ? 1ul : 0ul;
  rng_state[0] = new_low;
  rng_state[1] = old_high + carry;
}
)msl",
                            kernel_name, static_cast<uint64_t>(delta));
  }

  HloModule* hlo_module_;
  const se::DeviceDescription& gpu_device_info_;
  BufferAssignment* buffer_assignment_;
  mlir::MLIRContext* mlir_context_;
  CallGraph* call_graph_;
  std::vector<gpu::GpuExecutable::ConstantInfo>* constants_;
  std::vector<gpu::GpuExecutable::GlobalInfo>* globals_;
  std::string* msl_blob_;
  NameUniquer* msl_function_name_uniquer_;
  absl::flat_hash_set<const HloConstantInstruction*> emitted_constants_;
  std::optional<std::string> rng_state_symbol_name_;
  gpu::ThunkIdGenerator thunk_id_generator_;
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
  // use SequentialHloOrdering off the resulting schedule for buffer assignment.
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

  // Reuse the generic GPU thunk sequence emitter for scheduled-computation
  // traversal and control-flow thunks. MetalThunkEmissionBackend handles the
  // target-specific pieces: constants, MLIR-to-MSL fusion emission, and RNG.
  mlir_context()->appendDialectRegistry(
      gpu::MlirKernelEmitter::GetDialectRegistry());
  mlir_context()->loadAllAvailableDialects();
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module.get());
  std::vector<gpu::GpuExecutable::ConstantInfo> constants;
  std::vector<gpu::GpuExecutable::GlobalInfo> globals;
  std::string msl_blob;
  NameUniquer msl_function_name_uniquer;
  MetalThunkEmissionBackend metal_thunk_backend(
      hlo_module.get(), gpu_device_info, buffer_assignment.get(),
      mlir_context(), call_graph.get(), &constants, &globals, &msl_blob,
      &msl_function_name_uniquer);
  gpu::ThunkSequenceEmitter thunk_emitter(&metal_thunk_backend);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<gpu::SequentialThunk> sequential_thunk,
      thunk_emitter.EmitHloEntryComputation(hlo_module.get()));

  TF_ASSIGN_OR_RETURN(auto output_info,
                      gpu::GetOutputInfo(*hlo_module, *buffer_assignment));

  ProgramShape program_shape =
      entry->ComputeProgramShape(/*include_ids=*/false);
  std::string module_name(hlo_module->name());

  gpu::GpuExecutable::Params params;
  params.executable =
      std::make_unique<gpu::ThunkExecutor>(
          std::move(sequential_thunk->thunks()));
  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "", "msl", msl_blob);
  }
  params.asm_text = std::move(msl_blob);
  params.constants = std::move(constants);
  params.globals = std::move(globals);
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
