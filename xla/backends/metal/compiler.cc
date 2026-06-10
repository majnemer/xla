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

#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/metal/codegen/msl_kernel_emitter.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/sort_emitter.h"
#include "xla/backends/metal/runtime/metal_kernel_artifact.h"
#include "xla/backends/metal/runtime/metal_kernel_thunk.h"
#include "xla/backends/metal/runtime/metal_triangular_solve_thunk.h"
#include "xla/backends/metal/transforms/expand_complex_triangular_solve.h"
#include "xla/codegen/emitters/kernel_arguments.h"
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
#include "xla/xla_data.pb.h"

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

// One bitonic-sort stage queued for Phase-2 MLIR emission. Mirrors
// DeferredFusion in structure: a name for diagnostics, the planner-produced
// stage description (xor_masks, tile_size, launch dims, kernel args, entry
// name), and the MetalKernelThunk this stage's PSO eventually fills.
struct DeferredSortStage {
  std::string sort_name;
  xla::metal::SortStageDescription stage;
  xla::metal::MetalKernelThunk* thunk;
};

}  // namespace

MetalCompiler::MetalCompiler()
    : xla::gpu::GpuCompiler(stream_executor::metal::kMetalPlatformId,
                            /*pointer_size=*/8) {}

absl::Status MetalCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& /*gpu_version*/,
    se::dnn::VersionInfo /*dnn_version*/,
    const se::SemanticVersion& /*toolkit_version*/,
    CompilationStats* compilation_stats) {
  // MPSMatrixSolveTriangular is F32-only. Expand complex-typed trsms into
  // matmul+select sequences. Sits at the right pipeline stage — after
  // CholeskyExpander has produced its trsms but before LayoutAssignment
  // runs, so the new ops get layouts assigned by the standard machinery.
  HloPassPipeline pipeline("metal_pre_layout_assignment", compilation_stats);
  pipeline.AddPass<MetalExpandComplexTriangularSolve>();
  return pipeline
      .Run(hlo_module,
           /*execution_threads=*/{HloInstruction::kMainExecutionThread})
      .status();
}

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
  std::vector<gpu::GpuExecutable::ConstantInfo> constants;
  // Backend-synthesized module globals (RNG state). Slices captured by
  // thunks point into this deque; it is moved whole into
  // Params::extra_allocations, which preserves element addresses.
  std::deque<BufferAllocation> extra_allocations;
  std::string msl_blob;
  NameUniquer msl_function_name_uniquer;
  std::vector<DeferredFusion> deferred_fusions;
  std::vector<DeferredSortStage> deferred_sort_stages;
  std::optional<BufferAllocation::Slice> rng_state_slice;
  // PSO compilation needs a live id<MTLDevice>; MetalCompiler rejects null
  // stream_exec at entry so the cast is safe here.
  void* metal_device =
      stream_executor::metal::GetMetalDeviceOpaque(stream_exec);
  if (metal_device == nullptr) {
    return absl::FailedPreconditionError(
        "MetalCompiler::CompileToBackendResult: stream executor is not a "
        "MetalExecutor or device handle is unavailable.");
  }

  // Mutually-recursive helpers: emit_instruction dispatches per opcode and
  // appends thunks to `thunks_out`. Control-flow opcodes (kWhile /
  // kConditional / kCall) recurse via emit_computation, which walks a
  // scheduled HloComputation and returns its ThunkSequence.
  std::function<absl::Status(const HloInstruction*, gpu::ThunkSequence*)>
      emit_instruction;
  std::function<absl::StatusOr<gpu::ThunkSequence>(const HloComputation*)>
      emit_computation;
  emit_computation =
      [&](const HloComputation* comp) -> absl::StatusOr<gpu::ThunkSequence> {
    gpu::ThunkSequence local;
    for (const HloInstruction* instr :
         hlo_module->schedule().sequence(comp).instructions()) {
      TF_RETURN_IF_ERROR(emit_instruction(instr, &local));
    }
    return local;
  };
  emit_instruction = [&](const HloInstruction* instr,
                         gpu::ThunkSequence* thunks_out) -> absl::Status {
    gpu::ThunkSequence& thunks = *thunks_out;
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

      case HloOpcode::kTriangularSolve: {
        // MetalExpandComplexTriangularSolve runs in
        // OptimizeHloConvolutionCanonicalization and rewrites complex trsms
        // into matmul+select. Real ADJOINT reaches the thunk and is folded
        // into TRANSPOSE there (A^H == A^T for real A).
        const auto* trsm = Cast<HloTriangularSolveInstruction>(instr);
        const TriangularSolveOptions& opts =
            trsm->triangular_solve_options();
        const HloInstruction* a = trsm->operand(0);
        const HloInstruction* b = trsm->operand(1);
        const Shape& a_shape = a->shape();
        const Shape& b_shape = b->shape();
        TF_RET_CHECK(a_shape.dimensions().size() >= 2);
        TF_RET_CHECK(a_shape.dimensions(a_shape.dimensions().size() - 2) ==
                     a_shape.dimensions(a_shape.dimensions().size() - 1))
            << "Triangular solve: A must be square";
        const int64_t rank = b_shape.dimensions().size();
        const int64_t m = a_shape.dimensions(a_shape.dimensions().size() - 1);
        const int64_t num_rhs =
            b_shape.dimensions(opts.left_side() ? rank - 1 : rank - 2);
        int64_t batch_size = 1;
        for (int64_t i = 0; i < rank - 2; ++i) {
          batch_size *= b_shape.dimensions(i);
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_slice,
                            buffer_assignment->GetUniqueSlice(a, {}));
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_slice,
                            buffer_assignment->GetUniqueSlice(b, {}));
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                            buffer_assignment->GetUniqueSlice(trsm, {}));
        // MPS solves in place; copy B → result if buffer assignment didn't
        // already alias them.
        if (b_slice != result_slice) {
          thunks.push_back(std::make_unique<gpu::DeviceToDeviceCopyThunk>(
              gpu::Thunk::ThunkInfo{},
              /*source_buffer=*/ShapedSlice{b_slice, b_shape},
              /*destination_buffer=*/ShapedSlice{result_slice, b_shape},
              /*mem_size=*/b_slice.size()));
        }
        thunks.push_back(std::make_unique<MetalTriangularSolveThunk>(
            gpu::Thunk::ThunkInfo{}, opts, a_shape.element_type(), a_slice,
            a_shape, result_slice, b_shape, batch_size, m, num_rhs));
        break;
      }

      case HloOpcode::kInfeed: {
        const auto* infeed = Cast<HloInfeedInstruction>(instr);
        std::vector<ShapedSlice> dest_slices;
        TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
            infeed->shape(),
            [&](const Shape& subshape,
                const ShapeIndex& index) -> absl::Status {
              if (subshape.IsTuple() || subshape.IsToken()) {
                return absl::OkStatus();
              }
              if (!subshape.IsArray()) {
                return Internal("Unexpected subshape for infeed '%s' at %s",
                                infeed->ToString(), index.ToString());
              }
              TF_ASSIGN_OR_RETURN(
                  BufferAllocation::Slice data,
                  buffer_assignment->GetUniqueSlice(infeed, index));
              dest_slices.push_back(ShapedSlice{data, subshape});
              return absl::OkStatus();
            }));
        thunks.push_back(std::make_unique<gpu::InfeedThunk>(
            gpu::Thunk::ThunkInfo{}, std::move(dest_slices)));
        break;
      }

      case HloOpcode::kOutfeed: {
        const auto* outfeed = Cast<HloOutfeedInstruction>(instr);
        const HloInstruction* source = outfeed->operand(0);
        std::vector<ShapedSlice> source_slices;
        TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
            source->shape(),
            [&](const Shape& subshape,
                const ShapeIndex& index) -> absl::Status {
              if (subshape.IsTuple()) return absl::OkStatus();
              if (!subshape.IsArray()) {
                return Internal(
                    "Unexpected subshape for outfeed source '%s' at %s",
                    source->ToString(), index.ToString());
              }
              TF_ASSIGN_OR_RETURN(
                  BufferAllocation::Slice data,
                  buffer_assignment->GetUniqueSlice(source, index));
              source_slices.push_back(ShapedSlice{data, subshape});
              return absl::OkStatus();
            }));
        thunks.push_back(std::make_unique<gpu::OutfeedThunk>(
            gpu::Thunk::ThunkInfo{}, std::move(source_slices)));
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

      case HloOpcode::kRngGetAndUpdateState: {
        const auto* rng = Cast<HloRngGetAndUpdateStateInstruction>(instr);
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice output_slice,
            buffer_assignment->GetUniqueSlice(rng, /*index=*/{}));

        // The state is a module global: a 16-byte constant-backed allocation
        // whose device buffer the Metal executor allocates at LoadModule from
        // the matching ConstantInfo and seeds from its content (the same
        // 0x7012395 initialiser CUDA bakes into its rng_state LLVM global —
        // see llvm_util.cc GetOrCreateVariableForRngState). MSL has no
        // module-scope globals, so the state crosses the launch boundary as
        // a kernel argument; making it a real BufferAllocation keeps it
        // inside the slice machinery that thunk dependency analysis,
        // command-buffer capture, and buffer tooling reason with. All RNG
        // ops in the module share one state allocation; each declares a
        // write on the slice, so ThunkExecutor serialises them in schedule
        // order.
        if (!rng_state_slice.has_value()) {
          constexpr int64_t kStateBytes = 2 * sizeof(uint64_t);
          const int64_t state_idx = buffer_assignment->Allocations().size() +
                                    extra_allocations.size();
          BufferAllocation& state_alloc =
              extra_allocations.emplace_back(state_idx, kStateBytes,
                                             /*color=*/0);
          state_alloc.set_constant(true);
          rng_state_slice = BufferAllocation::Slice(&state_alloc, /*offset=*/0,
                                                    kStateBytes);

          std::array<uint64_t, 2> initial_state = {0x7012395ULL, 0};
          std::vector<uint8_t> content(sizeof(initial_state));
          std::memcpy(content.data(), initial_state.data(), content.size());

          gpu::GpuExecutable::ConstantInfo info;
          info.symbol_name = llvm_ir::SanitizeFunctionName(absl::StrCat(
              hlo_module->name(), "_", hlo_module->unique_id(),
              "_rng_state"));
          info.content = gpu::DenseDataIntermediate::Own(std::move(content));
          info.allocation_index = state_idx;
          constants.push_back(std::move(info));
        }

        // Per-op single-thread kernel with delta baked in. u128 increment as
        // two ulongs with carry.
        std::string entry_name = msl_function_name_uniquer.GetUniqueName(
            llvm_ir::SanitizeFunctionName(std::string(rng->name())));
        std::string rng_msl = absl::Substitute(
            R"msl(#include <metal_stdlib>
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
            entry_name, static_cast<uint64_t>(rng->delta()));
        if (!msl_blob.empty()) msl_blob.append("\n");
        msl_blob.append(rng_msl);

        TF_ASSIGN_OR_RETURN(
            stream_executor::metal::CompiledPipeline compiled,
            stream_executor::metal::CompileAndProbe(
                metal_device, rng_msl, entry_name,
                /*descriptor_thread_hint=*/1));

        const Shape kStateShape = ShapeUtil::MakeShape(U64, {2});
        std::vector<emitters::KernelArgument> kernel_arg_vec;
        kernel_arg_vec.emplace_back(kStateShape, *rng_state_slice);
        kernel_arg_vec.back().set_written(true);
        kernel_arg_vec.emplace_back(kStateShape, output_slice);
        kernel_arg_vec.back().set_written(true);
        emitters::KernelArguments kernel_args(std::move(kernel_arg_vec));

        auto thunk = std::make_unique<xla::metal::MetalKernelThunk>(
            gpu::Thunk::ThunkInfo{}, kernel_args);
        gpu::LaunchDimensions launch_dims(/*num_blocks=*/1,
                                          /*num_threads_per_block=*/1);
        auto artifact = std::make_unique<xla::metal::MetalKernelArtifact>(
            entry_name, /*arity=*/2, launch_dims, std::move(compiled.pso));
        thunk->SetArtifact(std::move(artifact));
        thunks.push_back(std::move(thunk));
        break;
      }

      case HloOpcode::kSort: {
        const auto* sort = Cast<HloSortInstruction>(instr);
        if (sort->is_stable()) {
          return Internal(
              "Metal: stable sort not supported here. Did stable_sort_expander "
              "run? Sort '%s'",
              sort->name());
        }
        // Sort runs in-place on the output buffer; copy non-iota, non-aliased
        // operands in first.
        for (int64_t i = 0; i < sort->operand_count(); ++i) {
          const HloInstruction* operand = sort->operand(i);
          if (HloPredicateIsOp<HloOpcode::kIota>(operand)) continue;
          ShapeIndex shape_index =
              sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
          TF_ASSIGN_OR_RETURN(
              BufferAllocation::Slice src,
              buffer_assignment->GetUniqueSlice(operand, {}));
          TF_ASSIGN_OR_RETURN(
              BufferAllocation::Slice dst,
              buffer_assignment->GetUniqueSlice(sort, shape_index));
          if (src == dst) continue;
          const Shape& shape = operand->shape();
          thunks.push_back(std::make_unique<gpu::DeviceToDeviceCopyThunk>(
              gpu::Thunk::ThunkInfo{},
              /*source_buffer=*/ShapedSlice{src, shape},
              /*destination_buffer=*/ShapedSlice{dst, shape},
              /*mem_size=*/src.size()));
        }
        std::string entry_name_prefix = msl_function_name_uniquer.GetUniqueName(
            llvm_ir::SanitizeFunctionName(std::string(sort->name())));
        TF_ASSIGN_OR_RETURN(
            std::vector<xla::metal::SortStageDescription> stages,
            xla::metal::PlanBitonicSort(sort, *buffer_assignment, gpu_device_info,
                                        gpu::GetDefaultBufferAlignment(),
                                        entry_name_prefix));
        for (auto& stage : stages) {
          auto stage_thunk = std::make_unique<xla::metal::MetalKernelThunk>(
              gpu::Thunk::ThunkInfo{}, stage.kernel_args);
          auto* stage_thunk_ptr = stage_thunk.get();
          thunks.push_back(std::move(stage_thunk));
          deferred_sort_stages.push_back(DeferredSortStage{
              /*sort_name=*/std::string(sort->name()),
              /*stage=*/std::move(stage),
              /*thunk=*/stage_thunk_ptr,
          });
        }
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

      case HloOpcode::kWhile: {
        TF_ASSIGN_OR_RETURN(
            gpu::ThunkSequence cond_thunks,
            emit_computation(instr->while_condition()));
        TF_ASSIGN_OR_RETURN(gpu::ThunkSequence body_thunks,
                            emit_computation(instr->while_body()));
        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice pred_slice,
            buffer_assignment->GetUniqueSlice(
                instr->while_condition()->root_instruction(), {}));
        std::optional<int64_t> trip_count;
        TF_ASSIGN_OR_RETURN(auto cfg,
                            instr->backend_config<WhileLoopBackendConfig>());
        if (cfg.has_known_trip_count()) {
          trip_count = cfg.known_trip_count().n();
        }
        thunks.push_back(std::make_unique<gpu::WhileThunk>(
            gpu::Thunk::ThunkInfo{}, pred_slice, std::move(cond_thunks),
            std::move(body_thunks), trip_count));
        break;
      }

      case HloOpcode::kConditional: {
        std::vector<gpu::ThunkSequence> branch_thunks;
        branch_thunks.reserve(instr->branch_count());
        for (HloComputation* branch : instr->branch_computations()) {
          TF_ASSIGN_OR_RETURN(gpu::ThunkSequence b,
                              emit_computation(branch));
          branch_thunks.push_back(std::move(b));
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice idx_slice,
                            buffer_assignment->GetUniqueSlice(
                                instr->operand(0), {}));
        thunks.push_back(std::make_unique<gpu::ConditionalThunk>(
            gpu::Thunk::ThunkInfo{},
            ShapedSlice{idx_slice, instr->operand(0)->shape()},
            std::move(branch_thunks)));
        break;
      }

      case HloOpcode::kCall: {
        // Inline the called computation's thunks directly; no separate Thunk
        // wrapper. Mirrors ThunkEmitter::EmitCallComputation.
        TF_RET_CHECK(instr->called_computations().size() == 1);
        TF_ASSIGN_OR_RETURN(
            gpu::ThunkSequence call_thunks,
            emit_computation(instr->called_computations().front()));
        for (std::unique_ptr<gpu::Thunk>& t : call_thunks) {
          thunks.push_back(std::move(t));
        }
        break;
      }

      default:
        return Unimplemented(
            "MetalCompiler::CompileToBackendResult: post-scheduling HLO "
            "opcode '%s' is not yet supported on Metal.",
            HloOpcodeString(instr->opcode()));
    }
    return absl::OkStatus();
  };

  TF_ASSIGN_OR_RETURN(gpu::ThunkSequence thunks, emit_computation(entry));

  // Phase 2: per-fusion retry loop. Each worker emits MLIR with a derived
  // device description that caps threads_per_block_limit at the current hint,
  // lowers, translates to MSL, then compiles + probes the PSO. If the PSO
  // grants fewer threads than the launch requests, we drop the hint to a
  // multiple of SIMD width at-or-below the PSO ceiling and retry. Floors at
  // one SIMD group; if even that won't host the kernel, surface a clean
  // ResourceExhaustedError naming the fusion. An emitter that ignores the
  // soft cap (num_threads_per_block stays put across retries) trips the
  // unchanged-requested-threads guard so we bail instead of spinning.
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
            "non-MLIR emitter (%s); only MLIR-kernel fusions are "
            "supported on Metal.",
            deferred.fusion_name, typeid(*emitter).name());
      }
      TF_ASSIGN_OR_RETURN(MlirKernelSource mlir_source,
                          mlir_fusion->mlir_kernel_emitter()->Emit(
                              context.get(), *deferred.fusion_instr,
                              deferred.entry_name, buffer_assignment.get()));
      gpu::LaunchDimensions launch_dims = mlir_fusion->launch_dimensions();
      int unroll_factor =
          mlir_fusion->mlir_kernel_emitter()->unroll_factor();

      mlir::ModuleOp module = mlir_source.module();
      TF_RETURN_IF_ERROR(metal::RunMetalLoweringPipeline(
          module, dev, unroll_factor, *hlo_module, deferred.entry_name,
          /*dump_category=*/"mlir-fusion"));
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

  // Phase 2/3 for sort stages. Sort kernels don't go through the fusion
  // MLIR-kernel-emitter; we build their MLIR modules directly from each
  // SortStageDescription. Retry is simpler than fusion's: tiled stages can
  // ShrinkSortStageTile on a PSO grant below requested threads; non-tiled
  // stages fall back to threads_per_block_limit just like fusions.
  auto build_sort_artifact =
      [&](xla::metal::SortStageDescription stage) -> absl::StatusOr<BuildResult> {
    const int64_t simd_width = gpu_device_info.threads_per_warp();
    for (;;) {
      auto context = std::make_unique<mlir::MLIRContext>();
      context->appendDialectRegistry(
          gpu::MlirKernelEmitter::GetDialectRegistry());
      context->loadAllAvailableDialects();

      TF_ASSIGN_OR_RETURN(
          mlir::OwningOpRef<mlir::ModuleOp> owning_module,
          xla::metal::EmitSortStageModule(context.get(), stage));
      mlir::ModuleOp module = *owning_module;

      TF_RETURN_IF_ERROR(metal::RunMetalLoweringPipeline(
          module, gpu_device_info, /*max_unroll_factor=*/0, *hlo_module,
          stage.entry_name, /*dump_category=*/"mlir-sort"));
      NameUniquer per_stage_uniquer;
      TF_ASSIGN_OR_RETURN(
          metal::MslKernelSource msl_source,
          metal::EmitMslKernel(module, &per_stage_uniquer, *hlo_module,
                               stage.entry_name));
      std::string msl_src = std::move(msl_source).source();

      TF_ASSIGN_OR_RETURN(
          stream_executor::metal::CompiledPipeline compiled,
          stream_executor::metal::CompileAndProbe(
              metal_device, msl_src, stage.entry_name,
              stage.launch_dimensions.num_threads_per_block()));
      int64_t requested = stage.launch_dimensions.num_threads_per_block();
      if (compiled.pso_max_threads >= requested) {
        unsigned arity =
            static_cast<unsigned>(stage.kernel_args.args().size());
        auto artifact = std::make_unique<MetalKernelArtifact>(
            stage.entry_name, arity, stage.launch_dimensions,
            std::move(compiled.pso));
        return BuildResult{std::move(artifact), std::move(msl_src)};
      }
      // PSO ceiling is below the launch's threads-per-block. For tiled
      // stages, the tile width sets threads_per_block — shrink and re-emit.
      // For non-tiled stages, the launch divides the iteration shape and
      // can't be re-tiled here, so surface a ResourceExhausted.
      if (xla::metal::ShrinkSortStageTile(stage, gpu_device_info)) {
        VLOG(1) << "Sort stage '" << stage.entry_name << "': requested "
                << requested << " threads, PSO ceiling "
                << compiled.pso_max_threads << "; shrinking tile to "
                << stage.tile_size << " and retrying.";
        continue;
      }
      return absl::ResourceExhaustedError(absl::StrCat(
          "Sort stage '", stage.entry_name, "': requested ", requested,
          " threads per threadgroup, but the device's PSO grants only ",
          compiled.pso_max_threads, " (>= SIMD width ", simd_width,
          " required). Tile cannot shrink further."));
    }
  };

  std::vector<absl::StatusOr<BuildResult>> sort_results(
      deferred_sort_stages.size());
  if (thread_pool) {
    absl::BlockingCounter counter(deferred_sort_stages.size());
    for (size_t i = 0; i < deferred_sort_stages.size(); ++i) {
      thread_pool.get_mutable()->Schedule([&, i] {
        sort_results[i] =
            build_sort_artifact(std::move(deferred_sort_stages[i].stage));
        counter.DecrementCount();
      });
    }
    counter.Wait();
  } else {
    for (size_t i = 0; i < deferred_sort_stages.size(); ++i) {
      sort_results[i] =
          build_sort_artifact(std::move(deferred_sort_stages[i].stage));
    }
  }
  for (size_t i = 0; i < deferred_sort_stages.size(); ++i) {
    TF_RETURN_IF_ERROR(sort_results[i].status());
  }
  for (size_t i = 0; i < deferred_sort_stages.size(); ++i) {
    BuildResult build_result = *std::move(sort_results[i]);
    deferred_sort_stages[i].thunk->SetArtifact(
        std::move(build_result.artifact));
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
  params.extra_allocations = std::move(extra_allocations);
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
