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
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
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
#include "xla/stream_executor/metal/metal_device_handle.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/metal/metal_pso_probe.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/maybe_owning.h"
#include "xla/util.h"

namespace xla {
namespace metal {
namespace {

// Mirror of gpu_llvm_compiler.cc's `MaybeOwningThreadPool` factory. Keeps the
// per-fusion lower+translate phase optionally parallel via either a caller-
// supplied thread pool or one we own for the duration of the compile.
using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  // Deadlock guard: we must not be called from inside the pool we plan to
  // schedule into, because BlockingCounter::Wait would block the only thread
  // that could service the scheduled tasks.
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

// Per-fusion state captured by the serial HLO walk. Phase 1 records only the
// bookkeeping (entry name, kernel arguments, the source HloFusionInstruction);
// MLIR emission, lowering, MSL translation, PSO compile and PSO retry all run
// in Phase 2 inside `BuildArtifact`. The MetalKernelThunk pointer lets Phase 2
// install the resulting artifact once compilation settles.
struct DeferredFusion {
  std::string fusion_name;
  const HloFusionInstruction* fusion_instr;
  std::string entry_name;                  // globally unique.
  emitters::KernelArguments kernel_args;
  xla::metal::MetalKernelThunk* thunk;     // owned by SequentialThunk.
};

// One bitonic-sort stage queued for Phase-2 MLIR emission. Mirrors
// DeferredFusion in structure: a name for diagnostics, the planner-produced
// stage description (xor_masks, tile_size, launch dims, kernel args, entry
// name), and the MetalKernelThunk this stage's PSO eventually fills.
struct DeferredSortStage {
  std::string sort_name;
  xla::metal::SortStageDescription stage;
  xla::metal::MetalKernelThunk* thunk;     // owned by SequentialThunk.
};

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
  MetalThunkEmissionBackend(
      HloModule* hlo_module, const se::DeviceDescription& gpu_device_info,
      BufferAssignment* buffer_assignment, CallGraph* call_graph,
      std::vector<gpu::GpuExecutable::ConstantInfo>* constants,
      std::vector<gpu::GpuExecutable::GlobalInfo>* globals,
      std::string* msl_blob, NameUniquer* msl_function_name_uniquer,
      std::vector<DeferredFusion>* deferred_fusions,
      std::vector<DeferredSortStage>* deferred_sort_stages)
      : hlo_module_(hlo_module),
        gpu_device_info_(gpu_device_info),
        buffer_assignment_(buffer_assignment),
        call_graph_(call_graph),
        constants_(constants),
        globals_(globals),
        msl_blob_(msl_blob),
        msl_function_name_uniquer_(msl_function_name_uniquer),
        deferred_fusions_(deferred_fusions),
        deferred_sort_stages_(deferred_sort_stages) {}

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
      case HloOpcode::kInfeed:
        return EmitInfeed(Cast<HloInfeedInstruction>(instr));
      case HloOpcode::kOutfeed:
        return EmitOutfeed(Cast<HloOutfeedInstruction>(instr));
      case HloOpcode::kRngGetAndUpdateState:
        return EmitRngGetAndUpdateState(
            Cast<HloRngGetAndUpdateStateInstruction>(instr));
      case HloOpcode::kSort:
        return EmitSort(Cast<HloSortInstruction>(instr));
      case HloOpcode::kFft:
        return EmitFft(Cast<HloFftInstruction>(instr));
      case HloOpcode::kTriangularSolve:
        return EmitTriangularSolve(
            Cast<HloTriangularSolveInstruction>(instr));
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
    TF_ASSIGN_OR_RETURN(
        emitters::KernelArguments kernel_args,
        emitters::KernelArguments::Create(*buffer_assignment_,
                                          gpu::GetDefaultBufferAlignment(),
                                          fusion_instr));
    // Reserve the MSL entry name on the shared NameUniquer up front, while
    // we're still serial. The translator (run in Phase 2) is told to treat
    // this as the MLIR function name; helpers inside the fusion's MSL are
    // prefixed with it so they can't collide across fusions either.
    std::string entry_name = msl_function_name_uniquer_->GetUniqueName(
        llvm_ir::SanitizeFunctionName(std::string(fusion_instr->name())));

    auto thunk = std::make_unique<xla::metal::MetalKernelThunk>(
        GetThunkInfo(fusion_instr), kernel_args);
    auto* thunk_ptr = thunk.get();
    gpu::ThunkSequence thunks;
    thunks.push_back(std::move(thunk));

    deferred_fusions_->push_back(DeferredFusion{
        /*fusion_name=*/std::string(fusion_instr->name()),
        /*fusion_instr=*/fusion_instr,
        /*entry_name=*/std::move(entry_name),
        /*kernel_args=*/std::move(kernel_args),
        /*thunk=*/thunk_ptr,
    });
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitInfeed(
      const HloInfeedInstruction* infeed) {
    // Infeed's output is a tuple of (data..., token). Collect each array
    // leaf's slice as the destination for the host→device copy.
    std::vector<ShapedSlice> dest_slices;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        infeed->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
          if (!subshape.IsArray()) {
            return Internal("Unexpected subshape for infeed '%s' at %s",
                            infeed->ToString(), index.ToString());
          }
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              buffer_assignment_->GetUniqueSlice(infeed, index));
          dest_slices.push_back(ShapedSlice{data, subshape});
          return absl::OkStatus();
        }));
    gpu::ThunkSequence thunks;
    thunks.push_back(std::make_unique<gpu::InfeedThunk>(GetThunkInfo(infeed),
                                                        std::move(dest_slices)));
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitOutfeed(
      const HloOutfeedInstruction* outfeed) {
    // Outfeed source is operand(0); each array leaf becomes a source slice.
    const HloInstruction* source = outfeed->operand(0);
    std::vector<ShapedSlice> source_slices;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        source->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.IsTuple()) return absl::OkStatus();
          if (!subshape.IsArray()) {
            return Internal("Unexpected subshape for outfeed source '%s' at %s",
                            source->ToString(), index.ToString());
          }
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              buffer_assignment_->GetUniqueSlice(source, index));
          source_slices.push_back(ShapedSlice{data, subshape});
          return absl::OkStatus();
        }));
    gpu::ThunkSequence thunks;
    thunks.push_back(std::make_unique<gpu::OutfeedThunk>(
        GetThunkInfo(outfeed), std::move(source_slices)));
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitTriangularSolve(
      const HloTriangularSolveInstruction* trsm) {
    const TriangularSolveOptions& opts = trsm->triangular_solve_options();
    // ADJOINT on complex types is handled by MetalExpandComplexTriangularSolve
    // (runs in OptimizeHloConvolutionCanonicalization), which rewrites the
    // entire trsm into matmul+select. ADJOINT on real types reaches here and
    // is treated as TRANSPOSE in the thunk, since A^H == A^T for real A.
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
                        buffer_assignment_->GetUniqueSlice(a, {}));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_slice,
                        buffer_assignment_->GetUniqueSlice(b, {}));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                        buffer_assignment_->GetUniqueSlice(trsm, {}));

    gpu::ThunkSequence thunks;
    // XLA's contract: B is read-only, the solve writes X into the result
    // buffer. MPS solves in-place; if buffer assignment didn't alias B with
    // result, emit a D2D copy first so the trsm's RHS == solution buffer is
    // already populated with B's data.
    if (b_slice != result_slice) {
      thunks.push_back(std::make_unique<gpu::DeviceToDeviceCopyThunk>(
          GetThunkInfo(trsm),
          /*source_buffer=*/ShapedSlice{b_slice, b_shape},
          /*destination_buffer=*/ShapedSlice{result_slice, b_shape},
          /*mem_size=*/b_slice.size()));
    }
    thunks.push_back(std::make_unique<MetalTriangularSolveThunk>(
        GetThunkInfo(trsm), opts, a_shape.element_type(), a_slice, a_shape,
        result_slice, b_shape, batch_size, m, num_rhs));
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitFft(const HloFftInstruction* fft) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice arg_slice,
        buffer_assignment_->GetUniqueSlice(fft->operand(0), {}));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                        buffer_assignment_->GetUniqueSlice(fft, {}));
    gpu::ThunkSequence thunks;
    thunks.push_back(std::make_unique<gpu::FftThunk>(
        GetThunkInfo(fft), fft->fft_type(), fft->fft_length(),
        /*input_buffer=*/arg_slice,
        /*output_buffer=*/dest_slice,
        /*input_shape=*/fft->operand(0)->shape(),
        /*output_shape=*/fft->shape()));
    return thunks;
  }

  absl::StatusOr<gpu::ThunkSequence> EmitSort(const HloSortInstruction* sort) {
    if (sort->is_stable()) {
      return Internal(
          "Metal: stable sort not supported here. Did stable_sort_expander "
          "run? Sort '%s'",
          sort->name());
    }

    // First, the per-operand in-place copy: sort runs on the output buffer,
    // so non-aliased operands need to be copied into it before the bitonic
    // sweep begins. Iota operands are emitted directly inside the first sort
    // kernel below (no preceding copy).
    gpu::ThunkSequence thunks;
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      const HloInstruction* operand = sort->operand(i);
      if (HloPredicateIsOp<HloOpcode::kIota>(operand)) continue;
      ShapeIndex shape_index =
          sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src,
                          buffer_assignment_->GetUniqueSlice(operand, {}));
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst,
                          buffer_assignment_->GetUniqueSlice(sort, shape_index));
      if (src == dst) continue;
      const Shape& shape = operand->shape();
      thunks.push_back(std::make_unique<gpu::DeviceToDeviceCopyThunk>(
          GetThunkInfo(sort),
          /*source_buffer=*/ShapedSlice{src, shape},
          /*destination_buffer=*/ShapedSlice{dst, shape},
          /*mem_size=*/src.size()));
    }

    // Plan the bitonic sweep: one SortStageDescription per kernel invocation.
    std::string entry_name_prefix = msl_function_name_uniquer_->GetUniqueName(
        llvm_ir::SanitizeFunctionName(std::string(sort->name())));
    TF_ASSIGN_OR_RETURN(
        std::vector<xla::metal::SortStageDescription> stages,
        xla::metal::PlanBitonicSort(sort, *buffer_assignment_, gpu_device_info_,
                                    gpu::GetDefaultBufferAlignment(),
                                    entry_name_prefix));

    for (auto& stage : stages) {
      auto thunk = std::make_unique<xla::metal::MetalKernelThunk>(
          GetThunkInfo(sort), stage.kernel_args);
      auto* thunk_ptr = thunk.get();
      thunks.push_back(std::move(thunk));
      deferred_sort_stages_->push_back(DeferredSortStage{
          /*sort_name=*/std::string(sort->name()),
          /*stage=*/std::move(stage),
          /*thunk=*/thunk_ptr,
      });
    }
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
  CallGraph* call_graph_;
  std::vector<gpu::GpuExecutable::ConstantInfo>* constants_;
  std::vector<gpu::GpuExecutable::GlobalInfo>* globals_;
  std::string* msl_blob_;
  NameUniquer* msl_function_name_uniquer_;
  std::vector<DeferredFusion>* deferred_fusions_;
  std::vector<DeferredSortStage>* deferred_sort_stages_;
  absl::flat_hash_set<const HloConstantInstruction*> emitted_constants_;
  std::optional<std::string> rng_state_symbol_name_;
  gpu::ThunkIdGenerator thunk_id_generator_;
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
                                      const CompileOptions& options,
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
  // Phase 1 (serial): per-fusion MLIR emission into per-fusion MLIRContexts,
  // recorded in `deferred_fusions`. RNG / constant chunks land directly in
  // `msl_blob` here. Phase 2 (parallel, below) lowers + translates each
  // deferred fusion. Phase 3 appends the per-fusion MSL chunks.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module.get());
  std::vector<gpu::GpuExecutable::ConstantInfo> constants;
  std::vector<gpu::GpuExecutable::GlobalInfo> globals;
  std::string msl_blob;
  NameUniquer msl_function_name_uniquer;
  std::vector<DeferredFusion> deferred_fusions;
  std::vector<DeferredSortStage> deferred_sort_stages;
  MetalThunkEmissionBackend metal_thunk_backend(
      hlo_module.get(), gpu_device_info, buffer_assignment.get(),
      call_graph.get(), &constants, &globals, &msl_blob,
      &msl_function_name_uniquer, &deferred_fusions, &deferred_sort_stages);
  gpu::ThunkSequenceEmitter thunk_emitter(&metal_thunk_backend);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<gpu::SequentialThunk> sequential_thunk,
      thunk_emitter.EmitHloEntryComputation(hlo_module.get()));

  // Phase 2: per-fusion retry loop. Each worker emits MLIR (with a derived
  // device description that caps threads_per_block_limit at the current
  // hint), lowers, translates to MSL, compiles + probes the PSO. If the PSO
  // reports a ceiling below the launch's requested threads, we drop the hint
  // and re-emit. The loop floors at the device's SIMD width — if the kernel
  // can't fit even one SIMD group's worth of threads, we surface a
  // ResourceExhaustedError naming the fusion.
  //
  // PSO compilation requires a live id<MTLDevice>; MetalCompiler already
  // rejects null stream_exec at entry so this is always available here.
  void* metal_device =
      stream_executor::metal::GetMetalDeviceOpaque(stream_exec);
  if (metal_device == nullptr) {
    return absl::FailedPreconditionError(
        "MetalCompiler::CompileToBackendResult: stream executor is not a "
        "MetalExecutor or device handle is unavailable.");
  }

  // Thread-pool sourcing follows gpu_llvm_compiler's pattern: explicit caller
  // pool > `--xla_gpu_force_compilation_parallelism` flag > serial.
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/hlo_module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/1);

  // Per-fusion artifact build: emit MLIR, lower, translate, compile PSO,
  // probe + retry. On success returns the artifact (PSO + launch dims) and
  // the MSL string (for `asm_text` dump consumers). On failure returns a
  // status naming the fusion.
  struct BuildResult {
    std::unique_ptr<MetalKernelArtifact> artifact;
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
      // Build a derived device description with the current cap; pass it to
      // HloFusionAnalysis so the reduction emitter (and any other emitter
      // that consults threads_per_block_limit) shapes num_threads_
      // accordingly.
      se::DeviceDescription dev = gpu_device_info;
      dev.set_threads_per_block_limit(hint);

      // Per-fusion MLIRContext: no cross-fusion contention, no cross-attempt
      // contamination (each retry starts fresh).
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
      auto* mlir_fusion =
          dynamic_cast<gpu::MlirKernelFusion*>(emitter.get());
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
      if (auto status = metal::RunMetalLoweringPipeline(
              module, dev, unroll_factor, *hlo_module, deferred.entry_name,
              "mlir-fusion");
          !status.ok()) {
        return absl::InternalError(absl::StrCat(
            "MetalCompiler::CompileToBackendResult: ", status.message(),
            " (fusion '", deferred.fusion_name, "')"));
      }
      NameUniquer per_fusion_uniquer;
      TF_ASSIGN_OR_RETURN(metal::MslKernelSource msl_source,
                          metal::EmitMslKernel(module, &per_fusion_uniquer,
                                               *hlo_module,
                                               deferred.entry_name));
      last_msl = std::move(msl_source).source();

      // Compile + probe the PSO with the same hint we used to constrain the
      // emitter. The descriptor hint also tells Metal "you can spend up to
      // this many threads' worth of registers per thread," which usually
      // raises pso.maxTotalThreadsPerThreadgroup back up to the hint.
      TF_ASSIGN_OR_RETURN(
          stream_executor::metal::CompiledPipeline compiled,
          stream_executor::metal::CompileAndProbe(metal_device, last_msl,
                                                  deferred.entry_name, hint));
      int64_t requested = launch_dims.num_threads_per_block();
      // If the emitter can't honor a smaller cap (i.e., launch dims didn't
      // shrink despite a lower hint), retry is hopeless. Surface a
      // ResourceExhausted instead of spinning. This is the expected path for
      // emitters like ColumnReductionFusion whose num_threads_ is hard-wired
      // to 1024 by the indexing-map algebra.
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
        auto artifact = std::make_unique<MetalKernelArtifact>(
            deferred.entry_name, arity, launch_dims,
            std::move(compiled.pso));
        return BuildResult{std::move(artifact), std::move(last_msl)};
      }
      // PSO won't host the launch. Drop the hint to a multiple of SIMD width
      // at or below the PSO's reported ceiling and retry. Using a SIMD
      // multiple keeps the threadgroup from leaving partial warps unused.
      int64_t next = (compiled.pso_max_threads / simd_width) * simd_width;
      if (next >= hint) {
        // Metal contradicted itself — don't spin.
        next = hint - simd_width;
      }
      VLOG(1) << "Fusion '" << deferred.fusion_name << "': requested "
              << requested << " threads, PSO ceiling "
              << compiled.pso_max_threads << "; retrying with hint " << next;
      hint = next;
    }
  };

  std::vector<absl::StatusOr<BuildResult>> results(deferred_fusions.size());
  if (thread_pool) {
    absl::BlockingCounter counter(deferred_fusions.size());
    for (size_t i = 0; i < deferred_fusions.size(); ++i) {
      thread_pool.get_mutable()->Schedule([&, i] {
        results[i] = build_artifact(
            deferred_fusions[i],
            /*initial_max_threads_per_block=*/
            gpu_device_info.threads_per_block_limit());
        counter.DecrementCount();
      });
    }
    counter.Wait();
  } else {
    for (size_t i = 0; i < deferred_fusions.size(); ++i) {
      results[i] = build_artifact(
          deferred_fusions[i],
          /*initial_max_threads_per_block=*/
          gpu_device_info.threads_per_block_limit());
    }
  }
  for (size_t i = 0; i < deferred_fusions.size(); ++i) {
    TF_RETURN_IF_ERROR(results[i].status());
  }

  // Phase 3: install artifacts on their thunks and concatenate MSL into
  // `asm_text` for debug-dump consumers. The runtime no longer reads the
  // MSL — MetalKernelThunk dispatches via the artifact's precompiled PSO.
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

      if (auto status = metal::RunMetalLoweringPipeline(
              module, gpu_device_info, /*max_unroll_factor=*/0, *hlo_module,
              stage.entry_name, "mlir-sort");
          !status.ok()) {
        return absl::InternalError(absl::StrCat(
            "MetalCompiler::CompileToBackendResult: ", status.message(),
            " (sort stage '", stage.entry_name, "')"));
      }
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
