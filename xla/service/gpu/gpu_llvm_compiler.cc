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

#include "xla/service/gpu/gpu_llvm_compiler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/cpu/nanort/nanort_client.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/transforms/estimate_cub_scan_scratch_size.h"
#include "xla/backends/gpu/transforms/estimate_cub_sort_scratch_size.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/gpu_aot_compilation_result.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/legacy_gpu_aot_compilation_result.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/thunk_emitter.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/error_handler.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/maybe_owning.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kGpuExecutablePtxMarker = "// GPU Executable\n";

using BorrowedMlirContext =
    ObjectPool<std::unique_ptr<mlir::MLIRContext>>::BorrowedObject;
using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                           void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);
  VLOG(5) << error_string;
}

void MergeModuleStatsInPlace(const ModuleStats& from, ModuleStats& to) {
  for (const auto& [name, kernel_stats] : from) {
    to[name].load_bytes_spilled += kernel_stats.load_bytes_spilled;
    to[name].store_bytes_spilled += kernel_stats.store_bytes_spilled;
  }
}

absl::StatusOr<xla::cpu::CompilationResultProto> GetCpuCompilationResult(
    const HloModuleProto& hlo_proto,
    xla::cpu::TargetMachineOptions cpu_target_machine_options) {
  xla::cpu::NanoRtClient client;
  XlaComputation computation(hlo_proto);
  Compiler::CompileOptions cpu_compile_options;
  cpu_compile_options.cpu_target_config.emplace(
      std::move(cpu_target_machine_options));

  ASSIGN_OR_RETURN(std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
                   client.Compile(computation));
  ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> result,
                   client.Export(executable.get()));
  xla::cpu::CpuAotCompilationResult* cpu_aot_compilation_result =
      tsl::down_cast<cpu::CpuAotCompilationResult*>(result.get());
  return cpu_aot_compilation_result->proto();
}

}  // namespace

GpuLLVMCompiler::GpuLLVMCompiler(se::Platform::Id platform_id,
                                 const char* target_triple,
                                 const char* data_layout)
    : GpuCompiler(platform_id,
                  llvm::DataLayout(data_layout)
                      .getPointerSize(/*default_address_space=*/0)),
      target_triple_(target_triple),
      data_layout_(data_layout),
      mlir_context_pool_([]() { return std::make_unique<mlir::MLIRContext>(); }) {}

void GpuLLVMCompiler::AddDeviceSpecificScratchSizePasses(
    HloPassPipeline* pipeline,
    const Compiler::GpuTargetConfig& gpu_target_config) {
  pipeline->AddPass<EstimateCubSortScratchSize>(
      gpu_target_config.platform_name);
  pipeline->AddPass<EstimateCubScanScratchSize>(
      gpu_target_config.platform_name);
}

absl::StatusOr<GpuLLVMCompiler::BackendCompileResult>
GpuLLVMCompiler::CompileSingleModule(
    const HloModuleConfig& module_config,
    const stream_executor::DeviceDescription& device_description,
    const HloModule* debug_module, llvm::Module* llvm_module, bool relocatable,
    std::optional<int> shard_number) {
  tsl::profiler::TraceMe traceme("CompileSingleModule");
  const DebugOptions& debug_options = module_config.debug_options();
  {
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "GpuLLVMCompiler - Running LLVM verifier for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown)")),
        VLOG_IS_ON(4) && debug_options.xla_enable_scoped_logging_timers());

    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);

    std::string err;
    llvm::raw_string_ostream err_stream(err);
    TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR "
           "lowering. Rerun with --xla_dump_to to get the IR"
        << (debug_module
                ? absl::StrCat(" and looks for files with name containing: *",
                               FilenameFor(*debug_module, "", ""), "*")
                : ".");
  }

  const std::string debug_name = debug_module ? debug_module->name() : "";
  XlaScopedFatalErrorHandler fatal_error_handler([&debug_name](
                                                     absl::string_view reason) {
    LOG(ERROR) << "LLVM Fatal Error while compiling target binary for module: "
               << debug_name << " Reason: " << reason;
  });

  ASSIGN_OR_RETURN(
      BackendCompileResult result,
      CompileTargetBinary(module_config, llvm_module, device_description,
                          relocatable, debug_module, shard_number));

  const bool should_dump = DumpingEnabledForHloModule(
      debug_module ? debug_module->name() : "", debug_options);
  if (should_dump) {
    if (debug_module) {
      llvm_ir::DumpIrIfEnabled(
          *debug_module, *llvm_module, /*optimized=*/true,
          shard_number.has_value() ? std::to_string(*shard_number) : "");
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }

  CallUserPostOptimizationHook(*llvm_module);
  CallUserAsmHook(result.asm_text);
  return result;
}

absl::StatusOr<std::unique_ptr<GpuExecutable>>
GpuLLVMCompiler::CompileToBackendResult(
    std::unique_ptr<HloModule> module, const GpuTopology& gpu_topology,
    const CompileOptions& options,
    se::StreamExecutor* absl_nullable stream_exec) {
  tsl::profiler::TraceMe traceme("GpuLLVMCompiler::CompileToBackendResult");

  BinaryMap dnn_compiled_graphs;
  if (stream_exec) {
    se::dnn::DnnSupport* dnn_support = stream_exec->AsDnn();
    TF_RET_CHECK(dnn_support != nullptr);
    RETURN_IF_ERROR(RunCudnnCompilerPasses(module.get(), *dnn_support,
                                           &dnn_compiled_graphs));
  }

  const DebugOptions& debug_opts = module->config().debug_options();
  const se::DeviceDescription& gpu_device_info =
      gpu_topology.gpu_target_config().device_description;

  llvm::LLVMContext llvm_context;
  ASSIGN_OR_RETURN(BorrowedMlirContext borrowed_context,
                   mlir_context_pool_.GetOrCreate());

  std::unique_ptr<GpuAliasInfo> alias_info = GetAliasInfo(gpu_device_info);
  RETURN_IF_ERROR(ScheduleAndVerify(module.get(), gpu_topology,
                                    alias_info.get(), borrowed_context->get())
                      .status());

  absl::string_view cache_path =
      debug_opts.xla_gpu_kernel_cache_file();
  const bool use_cache = !cache_path.empty();

  CompileModuleResults compile_module_results;
  ModuleStats module_stats;
  absl::Mutex module_stats_m_;
  std::atomic<int> shard_number = 0;

  {
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock(
        GetLLVMCommandLineOptions(debug_opts));
    BufferValue::SizeFunction buffer_size_bytes_function =
        BufferSizeBytesFunction();

    auto llvm_compiler =
        [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
            const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
      ASSIGN_OR_RETURN(
          BackendCompileResult result,
          CompileSingleModule(module->config(), descr, module.get(),
                              &llvm_module, /*relocatable=*/false,
                              shard_number.fetch_add(1)));
      absl::MutexLock lock(module_stats_m_);
      MergeModuleStatsInPlace(result.module_stats, module_stats);
      return std::move(result.binary);
    };
    CubinCustomKernelCompiler kernel_compiler(
        std::move(llvm_compiler),
        gpu_topology.gpu_target_config().device_description, debug_opts,
        /*thread_pool=*/nullptr);
    kernel_compiler.SetPreOptimizationHook([&](const llvm::Module& module) {
      CallUserPreOptimizationHook(module);
    });

    xla::cpu::TargetMachineOptions cpu_target_machine_options =
        gpu_topology.host_target_machine_options().value();

    ASSIGN_OR_RETURN(
        compile_module_results,
        CompileModuleToLlvmIr(
            module.get(), &llvm_context, target_triple_, data_layout_,
            PlatformId(), gpu_topology, alias_info.get(),
            std::move(buffer_size_bytes_function), llvm_options_lock,
            &kernel_compiler, std::move(cpu_target_machine_options),
            &mlir_context_pool_));
  }

  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_ir::DumpIrIfEnabled(*module,
                             *compile_module_results.llvm_module_constants,
                             /*optimized=*/false, "constants");
    CallUserPreOptimizationHook(*compile_module_results.llvm_module_constants);
  }

  BackendCompileResult backend_result;
  ASSIGN_OR_RETURN(
      backend_result,
      CompileSingleModule(
          module->config(), gpu_device_info, module.get(),
          &*compile_module_results.llvm_module_constants,
          /*relocatable=*/false,
          /*shard_number=*/shard_number.fetch_add(1)));

  if (use_cache) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(cache_path, resolved_path)) {
      return FailedPrecondition("File path can not be resolved: %s",
                                cache_path);
    }
    const bool cache_file_exists =
        tsl::Env::Default()->FileExists(resolved_path).ok();
    const CompilationCacheProto& current_cache =
        compile_module_results.kernel_compilation_cache;
    RETURN_IF_ERROR(UpdateDiskKernelCache(resolved_path,
                                          /*do_append=*/cache_file_exists,
                                          current_cache));
  }

  {
    absl::MutexLock lock(module_stats_m_);
    MergeModuleStatsInPlace(module_stats, backend_result.module_stats);
  }
  if (!backend_result.asm_text.empty()) {
    backend_result.asm_text =
        absl::StrCat(kGpuExecutablePtxMarker, backend_result.asm_text);
  }

  RecordXlaDeviceBinarySize(backend_result.binary.size());
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        compile_module_results.executable->ToString(/*indent=*/0));
  }

  compile_module_results.executable->Walk([&](Thunk* thunk) {
    if (thunk->kind() == Thunk::Kind::kHostExecuteStart) {
      auto* host_execute_start_thunk =
          tsl::down_cast<HostExecuteStartThunk*>(thunk);
      absl::StatusOr<xla::cpu::CompilationResultProto> cpu_compilation_result =
          GetCpuCompilationResult(
              host_execute_start_thunk->executable_proto().hlo_module(),
              gpu_topology.host_target_machine_options().value());
      CHECK_OK(cpu_compilation_result) << "Failed to compile host executable.";
      *host_execute_start_thunk->mutable_executable_proto()
           ->mutable_aot_compilation_result() =
          std::move(cpu_compilation_result.value());
      CHECK_OK(host_execute_start_thunk->LoadExecutable())
          << "Failed to load host executable.";
    }
  });

  bool embed_ir_in_executable = debug_opts.xla_embed_ir_in_executable();
  bool embed_debug_info = debug_opts.xla_gpu_executable_embed_debug_info();

  ASSIGN_OR_RETURN(stream_executor::ExecutableAbiVersion executable_abi_version,
                   stream_executor::ExecutableAbiVersion::FromDeviceDescription(
                       gpu_device_info));

  ModuleStats final_module_stats = backend_result.module_stats;
  ASSIGN_OR_RETURN(
      std::unique_ptr<GpuExecutable> gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/embed_debug_info ? std::move(backend_result.asm_text)
                                        : std::string(),
          /*binary=*/std::move(backend_result.binary),
          /*dnn_compiled_graphs=*/std::move(dnn_compiled_graphs),
          /*executable=*/
          std::make_unique<ThunkExecutor>(
              std::move(compile_module_results.executable->thunks())),
          /*constants=*/std::move(compile_module_results.constants),
          /*output_info=*/std::move(compile_module_results.output_info),
          /*module_name=*/std::move(compile_module_results.module_name),
          /*program_shape=*/
          module->compute_computation_layout().ComputeProgramShape(),
          /*mlir_allocations=*/
          (compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(compile_module_results.allocations)),
          /*buffer_assignment=*/
          std::move(compile_module_results.buffer_assignment),
          /*alias_info=*/std::move(alias_info),
          /*debug_options=*/debug_opts,
          /*device_description=*/gpu_device_info,
          /*debug_module=*/options.embed_hlo_module
              ? std::move(module)
              : std::unique_ptr<HloModule>(),
          /*enable_debug_info_manager=*/embed_debug_info,
          /*module_stats=*/std::move(final_module_stats),
          /*executable_abi_version=*/executable_abi_version,
          /*cpu_target_machine_options=*/
          options.cpu_target_config.has_value()
              ? options.cpu_target_config->cpu_target_machine_options
              : std::nullopt}));

  if (embed_ir_in_executable &&
      compile_module_results.llvm_module_constants != nullptr) {
    std::string ir_module_string_before_opt = llvm_ir::DumpToString(
        *compile_module_results.llvm_module_constants);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }

  IncrementCompiledProgramsCount();

  if (embed_debug_info && gpu_executable->has_module()) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_buffer_assignment() =
        gpu_executable->buffer_assignment()->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
  }

  return gpu_executable;
}

absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
GpuLLVMCompiler::LegacyCompileAheadOfTime(
    std::unique_ptr<HloModule> hlo_module,
    const AotCompilationOptions& options) {
  CompileOptions compile_options;
  compile_options.device_allocator = options.device_allocator();
  compile_options.gpu_topology = options.gpu_topology();

  // Compile to a GpuExecutable, then export it.
  ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      RunBackend(std::move(hlo_module), options.executor(), compile_options));

  std::vector<std::unique_ptr<CompiledModule>> results;
  ASSIGN_OR_RETURN(results.emplace_back(), Export(executable.get()));
  return results;
}

absl::StatusOr<std::unique_ptr<CompiledModule>> GpuLLVMCompiler::Export(
    Executable* executable) {
  auto* gpu_executable = tsl::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) {
    return Internal("GpuExecutable is null");
  }
  if (gpu_executable->module()
          .config()
          .debug_options()
          .xla_gpu_experimental_aot_compiled_thunks()) {
    ASSIGN_OR_RETURN(GpuExecutableProto proto, gpu_executable->ToProto());
    return GpuAotCompilationResult::FromProto(std::move(proto));
  }
  return LegacyGpuAotCompilationResult::FromModule(
      &gpu_executable->module(), gpu_executable->buffer_assignment()->ToProto(),
      gpu_executable->text(), gpu_executable->binary(),
      gpu_executable->dnn_compiled_graphs(), GetPointerSize(), this);
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
GpuLLVMCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  auto reader =
      std::make_unique<riegeli::StringReader<>>(serialized_aot_result);
  ASSIGN_OR_RETURN(bool is_split_proto, IsSplitProto(*reader));
  if (is_split_proto) {
    return GpuAotCompilationResult::FromSerialized(std::move(reader));
  }
  GpuExecutableProto gpu_executable_proto;
  if (!gpu_executable_proto.ParseFromString(serialized_aot_result)) {
    return InvalidArgument(
        "Failed to parse serialized AOT result as GpuExecutableProto.");
  }
  return LegacyGpuAotCompilationResult::FromProto(gpu_executable_proto,
                                                  GetPointerSize(), this);
}

absl::StatusOr<std::unique_ptr<Executable>>
GpuLLVMCompiler::LoadExecutableFromAotResult(
    const CompiledModule& aot_result,
    const se::DeviceDescription& device_description) {
  tsl::profiler::TraceMe traceme("LoadExecutableFromAotResult");

  const auto* gpu_aot_result =
      dynamic_cast<const LegacyGpuAotCompilationResult*>(&aot_result);
  if (gpu_aot_result == nullptr) {
    return Internal(
        "AotCompilationResult is not a GpuThunkAotCompilationResult.");
  }
  const GpuExecutableProto& proto = gpu_aot_result->GetGpuExecutableProto();

  ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));

  ExecutionStreamAssignment execution_stream_assignment(
      hlo_module.get(),
      {
          kDefaultNumComputeStreams,
          hlo_module->config()
                  .debug_options()
                  .xla_gpu_experimental_enable_collective_multi_streaming()
              ? kDefaultNumCommunicationStreams
              : 1,
      });

  std::vector<uint8_t> binary(proto.binary().begin(), proto.binary().end());
  absl::string_view platform_name = PlatformId()->ToName();

  llvm::LLVMContext llvm_context;
  std::unique_ptr<GpuAliasInfo> alias_info = GetAliasInfo(device_description);
  ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto.buffer_assignment(), hlo_module.get(),
                                  BufferSizeBytesFunction(), alias_info.get()));

  std::atomic<int> shard_number = 0;
  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    ASSIGN_OR_RETURN(
        BackendCompileResult result,
        CompileSingleModule(hlo_module->config(), descr, hlo_module.get(),
                            &llvm_module, false, shard_number.fetch_add(1)));
    return std::move(result.binary);
  };
  CubinCustomKernelCompiler kernel_compiler(
      std::move(llvm_compiler), device_description,
      hlo_module->config().debug_options());
  kernel_compiler.SetPreOptimizationHook(
      [&](const llvm::Module& module) { CallUserPreOptimizationHook(module); });

  ASSIGN_OR_RETURN(BorrowedMlirContext borrowed_context,
                   mlir_context_pool_.GetOrCreate());

  IrEmitterContext ir_emitter_context(
      hlo_module.get(), buffer_assignment.get(), &execution_stream_assignment,
      platform_name, device_description, borrowed_context->get(), &llvm_context,
      llvm::Triple(target_triple()), data_layout(), &kernel_compiler,
      cpu::TargetMachineOptions(hlo_module->config().debug_options()),
      &mlir_context_pool_);

  absl::string_view cache_file_path =
      hlo_module->config().debug_options().xla_gpu_kernel_cache_file();
  if (!cache_file_path.empty()) {
    RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }

  xla::llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock(
      GetLLVMCommandLineOptions(hlo_module->config().debug_options()));

  ThunkEmitter thunk_emitter(&ir_emitter_context, &llvm_options_lock);
  ASSIGN_OR_RETURN(auto sequential_thunk,
                   thunk_emitter.EmitHloEntryComputation(hlo_module.get()));

  std::vector<GpuExecutable::ConstantInfo> constants =
      std::move(ir_emitter_context.constants());
  ASSIGN_OR_RETURN(auto output_info,
                   GetOutputInfo(*hlo_module, *buffer_assignment));
  ProgramShape program_shape =
      hlo_module->entry_computation_layout().ComputeProgramShape();
  *program_shape.mutable_result() = hlo_module->result_shape();
  DebugOptions debug_options = hlo_module->config().debug_options();
  std::string hlo_module_name = hlo_module->name();

  ASSIGN_OR_RETURN(auto executable_abi_version,
                   stream_executor::ExecutableAbiVersion::FromDeviceDescription(
                       device_description));

  std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options =
      std::nullopt;
  if (proto.has_cpu_target_machine_options()) {
    ASSIGN_OR_RETURN(cpu_target_machine_options,
                     xla::cpu::TargetMachineOptions::FromProto(
                         proto.cpu_target_machine_options()));
  }
  BufferAssignmentProto buffer_assignment_proto = buffer_assignment->ToProto();
  tsl::profiler::TraceMe create_traceme("CreateGpuExecutable");
  return GpuExecutable::Create(GpuExecutable::Params{
      /*asm_text=*/proto.asm_text(),
      /*binary=*/binary,
      /*dnn_compiled_graphs=*/
      BinaryMap(proto.dnn_compiled_graphs().cbegin(),
                proto.dnn_compiled_graphs().cend()),
      /*executable=*/
      std::make_unique<ThunkExecutor>(std::move(sequential_thunk->thunks())),
      /*constants=*/std::move(constants),
      /*output_info=*/std::move(output_info),
      /*module_name=*/std::move(hlo_module_name),
      /*program_shape=*/std::move(program_shape),
      /*mlir_allocations=*/std::move(*buffer_assignment).TakeAllocations(),
      /*buffer_assignment=*/nullptr,
      /*alias_info=*/std::move(alias_info),
      /*debug_options=*/std::move(debug_options),
      /*device_description=*/device_description,
      /*debug_module=*/std::move(hlo_module),
      /*enable_debug_info_manager=*/true,
      /*module_stats=*/{},
      /*executable_abi_version=*/executable_abi_version,
      /*cpu_target_machine_options=*/std::move(cpu_target_machine_options),
      /*buffer_assignment_proto=*/std::move(buffer_assignment_proto),
  });
}

}  // namespace gpu
}  // namespace xla
