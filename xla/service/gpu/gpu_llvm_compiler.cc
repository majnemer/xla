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
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/cpu/nanort/nanort_client.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/estimate_cub_scan_scratch_size.h"
#include "xla/backends/gpu/transforms/estimate_cub_sort_scratch_size.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/execution_stream_assignment.h"
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
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/error_handler.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/maybe_owning.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kGpuExecutablePtxMarker = "// GPU Executable\n";

using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  tsl::profiler::TraceMe traceme("CreateMaybeOwningThreadPool");
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

void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                           void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  VLOG(5) << error_string;
}

std::string SingleFunctionName(const llvm::Module& module) {
  std::string name;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      if (name.empty()) {
        name = func.getName().str();
      } else {
        return "";
      }
    }
  }
  return name;
}

absl::StatusOr<int32_t> GetNumDevicesFromPlatform(se::PlatformId platform_id) {
  absl::StatusOr<se::Platform*> platform =
      se::PlatformManager::PlatformWithId(platform_id);
  if (!platform.ok()) {
    return absl::Status(
        platform.status().code(),
        absl::StrCat(
            platform.status().message(),
            ". Are you missing gpu_plugin or stream_executor dependency?"));
  }
  return platform.value()->VisibleDeviceCount();
}

absl::StatusOr<GpuTopology> InferGpuTopology(
    const HloModuleConfig& hlo_config,
    se::StreamExecutor* absl_nullable stream_exec,
    const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
    se::PlatformId platform_id) {
  int32_t num_partitions;
  int32_t num_hosts_per_partition;
  int32_t num_devices_per_host;
  std::optional<GpuTargetConfig> gpu_target_config;
  std::optional<cpu::TargetMachineOptions> cpu_target_options;

  if (options.cpu_target_config.has_value()) {
    cpu_target_options = options.cpu_target_config->cpu_target_machine_options;
  }

  if (options.gpu_topology.has_value()) {
    const GpuTopology& gpu_topology = *options.gpu_topology;
    if (gpu_topology.has_gpu_target_config()) {
      gpu_target_config = gpu_topology.gpu_target_config();
    }
    num_partitions = gpu_topology.num_partitions();
    num_hosts_per_partition = gpu_topology.num_hosts_per_partition();
    num_devices_per_host = gpu_topology.num_devices_per_host();
    if (gpu_topology.host_target_machine_options().has_value()) {
      cpu_target_options = gpu_topology.host_target_machine_options();
    }
  } else {
    num_partitions = hlo_config.num_partitions();
    num_hosts_per_partition = 1;
    ASSIGN_OR_RETURN(num_devices_per_host,
                     GetNumDevicesFromPlatform(platform_id));
  }

  if (!gpu_target_config.has_value() &&
      !debug_opts.xla_gpu_target_config_filename().empty()) {
    ASSIGN_OR_RETURN(
        gpu_target_config,
        GetTargetConfigFromFile(debug_opts.xla_gpu_target_config_filename()));
  }

  if (!gpu_target_config.has_value() && stream_exec == nullptr) {
    return absl::InvalidArgumentError(
        "Couldn't determine the target compilation environment. Either stream "
        "executor (GPU) has to be attached for JIT compilation, or a target "
        "config has to be passed in as a parameter or provided via "
        "--xla_gpu_target_config_filename for AOT compilation.");
  }

  if (!cpu_target_options.has_value()) {
    VLOG(2) << "Inferring CPU target options from the host architecture.";
    cpu_target_options.emplace(debug_opts);
  }

  if (gpu_target_config.has_value()) {
    VLOG(2) << "Found target compilation environment, and "
            << (stream_exec == nullptr
                    ? "not stream executor. Performing deviceless compilation."
                    : "stream executor. Performing cross compilation.");
    return GpuTopology{gpu_target_config->device_description.platform_version(),
                       num_partitions,
                       num_hosts_per_partition,
                       num_devices_per_host,
                       std::move(gpu_target_config),
                       std::move(cpu_target_options)};
  }

  VLOG(1) << "Found stream executor, and not a target compilation environment. "
             "Performing JIT compilation.";
  GpuTargetConfig local_target_config = GpuTargetConfig{stream_exec};

  int64_t device_memory_size =
      local_target_config.device_description.device_memory_size();
  if (device_memory_size == -1) {
    return absl::FailedPreconditionError(
        "When running on an NVIDIA simulation device, you must use "
        "--xla_gpu_target_config_filename to pass in target information. "
        "The target config from StreamExecutor is inaccurate.");
  }
  return GpuTopology{stream_exec->GetDeviceDescription().platform_version(),
                     num_partitions,
                     num_hosts_per_partition,
                     num_devices_per_host,
                     std::move(local_target_config),
                     std::move(cpu_target_options)};
}

absl::StatusOr<xla::cpu::CompilationResultProto> GetCpuCompilationResult(
    const HloModuleProto& hlo_proto,
    xla::cpu::TargetMachineOptions cpu_target_machine_options) {
  xla::cpu::NanoRtClient client;
  XlaComputation computation(hlo_proto);
  Compiler::CompileOptions cpu_compile_options;
  cpu_compile_options.cpu_target_config.emplace(
      std::move(cpu_target_machine_options));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
                      client.Compile(computation));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> result,
                      client.Export(executable.get()));
  xla::cpu::CpuAotCompilationResult* cpu_aot_compilation_result =
      tsl::down_cast<xla::cpu::CpuAotCompilationResult*>(result.get());
  return cpu_aot_compilation_result->proto();
}

}  // namespace

GpuLLVMCompiler::GpuLLVMCompiler(se::Platform::Id platform_id,
                                 const char* target_triple,
                                 const char* data_layout)
    : GpuCompiler(platform_id,
                  llvm::DataLayout(data_layout)
                      .getPointerSize(0 /* default address space */)),
      target_triple_(target_triple),
      data_layout_(data_layout),
      mlir_context_pool_([]() { return CreateMlirContext(); }) {}

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
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "GpuCompiler::RunBackend - Running LLVM verifier for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown)")),
        VLOG_IS_ON(4) && debug_options.xla_enable_scoped_logging_timers());

    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);

    std::string err;
    llvm::raw_string_ostream err_stream(err);

    // verifyModule() returns true if the module is broken.
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
          *debug_module, *llvm_module,
          /*optimized=*/true,
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

absl::StatusOr<GpuLLVMCompiler::BackendCompileResult>
GpuLLVMCompiler::CompileAndLink(
    const HloModuleConfig& module_config,
    CompileModuleResults& compile_module_results,
    const se::DeviceDescription& device_description,
    const CompileOptions& options, const HloModule* debug_module,
    se::StreamExecutor* stream_exec) {
  tsl::profiler::TraceMe traceme("CompileAndLink");

  absl::string_view cache_path =
      module_config.debug_options().xla_gpu_kernel_cache_file();
  const bool use_cache = !cache_path.empty();

  struct NamedModule {
    // The string is the function name for single-function modules (used to
    // cache them), empty for all other modules.
    std::string name;
    llvm::Module* module;
  };
  std::vector<NamedModule> llvm_modules;
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/module_config.debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/1);
  // Only single-function module are cacheable -> for caching try to get 1
  // function per module.

  absl::flat_hash_set<std::string> compiled_functions;
  llvm_modules.reserve(compile_module_results.llvm_modules.size() + 1);

  int single_function_module_count = 0;
  for (std::unique_ptr<llvm::Module>& module :
       compile_module_results.llvm_modules) {
    const std::string name = SingleFunctionName(*module);
    if (!name.empty()) {
      ++single_function_module_count;
    }
    llvm_modules.push_back({name, module.get()});
    compiled_functions.insert(name);
  }
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_modules.push_back(
        {"", compile_module_results.llvm_module_constants.get()});
  }

  VLOG(2) << "Single-function cacheable modules: "
          << single_function_module_count << " / " << llvm_modules.size();

  struct NamedCompileResult {
    // Single function name or empty just like for llvm_modules.
    std::string name;
    absl::StatusOr<BackendCompileResult> result;
  };
  std::vector<NamedCompileResult> compile_results(llvm_modules.size());
  if (thread_pool) {
    absl::BlockingCounter counter(llvm_modules.size());
    for (int i = 0; i < llvm_modules.size(); ++i) {
      thread_pool.get_mutable()->Schedule([&compile_results, i, &llvm_modules,
                                           &counter, this, &module_config,
                                           &device_description, &debug_module] {
        // Each thread has its own context to avoid race conditions.
        llvm::LLVMContext new_context;
        std::unique_ptr<llvm::Module> new_module =
            CopyToContext(*llvm_modules.at(i).module, new_context);
        compile_results.at(i) = {
            llvm_modules.at(i).name,
            CompileSingleModule(module_config, device_description, debug_module,
                                new_module.get(),
                                /*relocatable=*/true,
                                /*shard_number=*/i)};
        counter.DecrementCount();
      });
    }
    counter.Wait();
  } else {
    for (int i = 0; i < llvm_modules.size(); ++i) {
      compile_results.at(i) = {
          llvm_modules.at(i).name,
          CompileSingleModule(module_config, device_description, debug_module,
                              &*llvm_modules.at(i).module,
                              /*relocatable=*/true,
                              /*shard_number=*/i)};
    }
  }

  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> binaries_to_link;
  binaries_to_link.reserve(compile_results.size());
  std::vector<KernelReuseCache::NamedBinary> binaries_to_cache;
  binaries_to_cache.reserve(single_function_module_count);
  for (const auto& [name, maybe_result] : compile_results) {
    ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.binary.empty()) {
      continue;
    }
    absl::StrAppend(&ptx_snippets, result.asm_text, "\n");
    binaries_to_link.push_back(result.binary);
    if (!name.empty()) {
      binaries_to_cache.push_back({name, result.binary});
    }
  }

  if (use_cache) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(cache_path, resolved_path)) {
      return FailedPrecondition("File path can not be resolved: %s",
                                cache_path);
    }
    const CompilationCacheProto& current_cache =
        compile_module_results.kernel_compilation_cache;
    const bool cache_file_exists =
        tsl::Env::Default()->FileExists(resolved_path).ok();
    if (cache_file_exists) {
      int loaded_kernel_count = 0;
      for (const auto& [name, entry] : current_cache.entries()) {
        if (compiled_functions.contains(name)) {
          VLOG(5) << "Using the just compiled kernel for " << name;
          TF_RET_CHECK(entry.binary().empty())
              << name
              << " is a just compiled kernel and is not expected to have a "
                 "binary yet.";
          continue;
        }
        const uint8_t* binary =
            reinterpret_cast<const uint8_t*>(entry.binary().data());
        if (entry.link_binary()) {
          binaries_to_link.push_back(
              std::vector<uint8_t>(binary, binary + entry.binary().size()));
        }
        VLOG(5) << "Using " << name << " from cache: " << entry.binary().size();
        ++loaded_kernel_count;
      }
      VLOG(2) << "Using " << loaded_kernel_count << " / "
              << current_cache.entries_size() << " cached kernels.";
    }
    RETURN_IF_ERROR(UpdateDiskKernelCache(resolved_path,
                                          /*do_append=*/cache_file_exists,
                                          current_cache, binaries_to_cache));
  }

  auto maybe_backend_result =
      LinkModules(device_description, std::move(binaries_to_link),
                  module_config.debug_options(), stream_exec);
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use XLA_FLAGS="
                  "--xla_gpu_enable_llvm_module_compilation_parallelism=false "
                  "to bypass it, but expect to get longer compilation time due "
                  "to the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }
  VLOG(4) << "Binary size after linking [B]: " << maybe_backend_result->size();
  compile_module_results.kernel_compilation_cache.Clear();
  return BackendCompileResult{ptx_snippets, std::move(*maybe_backend_result)};
}

absl::StatusOr<GpuLLVMCompiler::CompileResultWithMetadata>
GpuLLVMCompiler::CompileToBackendResultImpl(
    HloModule* module, llvm::LLVMContext* llvm_context,
    const GpuTopology& gpu_topology, const CompileOptions& options,
    se::StreamExecutor* stream_exec) {
  tsl::profiler::TraceMe traceme("CompileToBackendResultImpl");
  std::unique_ptr<GpuAliasInfo> alias_info =
      GetAliasInfo(gpu_topology.gpu_target_config().device_description);
  TF_ASSIGN_OR_RETURN(ScheduleMetadata schedule_metadata,
                      ScheduleAndVerify(module, gpu_topology, alias_info.get()));
  (void)schedule_metadata;

  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/tsl::port::MaxParallelism());

  ASSIGN_OR_RETURN(
      bool can_use_link_modules,
      CanUseLinkModules(module->config(),
                        gpu_topology.gpu_target_config().device_description,
                        stream_exec));
  const bool split_modules =
      can_use_link_modules &&
      module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism();

  CompileModuleResults compile_module_results;
  std::atomic<int> shard_number = 0;

  {
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock(
        GetLLVMCommandLineOptions(module->config().debug_options()));
    BufferValue::SizeFunction buffer_size_bytes_function =
        BufferSizeBytesFunction();

    auto llvm_compiler =
        [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
            const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
      ASSIGN_OR_RETURN(
          BackendCompileResult result,
          CompileSingleModule(module->config(), descr, module, &llvm_module,
                              false, shard_number.fetch_add(1)));
      return std::move(result.binary);
    };
    CubinCustomKernelCompiler kernel_compiler(
        std::move(llvm_compiler),
        gpu_topology.gpu_target_config().device_description,
        module->config().debug_options(), thread_pool.get_mutable());
    kernel_compiler.SetPreOptimizationHook([&](const llvm::Module& module) {
      CallUserPreOptimizationHook(module);
    });

    // Compile the module to thunks and llvm IR.
    xla::cpu::TargetMachineOptions cpu_target_machine_options =
        gpu_topology.host_target_machine_options().value();

    ASSIGN_OR_RETURN(
        compile_module_results,
        CompileModuleToLlvmIr(
            module, llvm_context, target_triple_, data_layout_, PlatformId(),
            gpu_topology, alias_info.get(),
            std::move(buffer_size_bytes_function), llvm_options_lock,
            &kernel_compiler, std::move(cpu_target_machine_options),
            &mlir_context_pool_));
  }

  for (const std::unique_ptr<llvm::Module>& llvm_module :
       compile_module_results.llvm_modules) {
    llvm_ir::DumpIrIfEnabled(*module, *llvm_module,
                             /*optimized=*/false,
                             std::to_string(shard_number.fetch_add(1)));
    CallUserPreOptimizationHook(*llvm_module);
  }
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_ir::DumpIrIfEnabled(*module,
                             *compile_module_results.llvm_module_constants,
                             /*optimized=*/false, "constants");
    CallUserPreOptimizationHook(*compile_module_results.llvm_module_constants);

    if (!can_use_link_modules) {
      compile_module_results.llvm_modules.push_back(
          std::move(compile_module_results.llvm_module_constants));
    }
  }

  BackendCompileResult backend_result;
  // Disable multi-threading during deviceless AOT compilation.
  // TODO(anlunx): Enable multi-threading once deviceless AOT compilation is
  // enabled.
  if (split_modules) {
    ASSIGN_OR_RETURN(
        backend_result,
        CompileAndLink(module->config(), compile_module_results,
                       gpu_topology.gpu_target_config().device_description,
                       options, module, stream_exec));
    LinkLlvmModulesInPlace(compile_module_results.llvm_modules);
  } else {
    LinkLlvmModulesInPlace(compile_module_results.llvm_modules);
    if (compile_module_results.llvm_module_constants) {
      std::vector<std::unique_ptr<llvm::Module>> modules;
      modules.push_back(std::move(compile_module_results.llvm_modules[0]));
      modules.push_back(
          std::move(compile_module_results.llvm_module_constants));
      LinkLlvmModulesInPlace(modules);
      compile_module_results.llvm_modules[0] = std::move(modules[0]);
    }
    ASSIGN_OR_RETURN(
        backend_result,
        CompileSingleModule(module->config(),
                            gpu_topology.gpu_target_config().device_description,
                            module, &*compile_module_results.llvm_modules[0],
                            /*relocatable=*/false,
                            /*shard_number=*/shard_number.fetch_add(1)));
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

  // Host executable has to be compiled the GPU compilation is done to
  // avoid a deadlock on the LLVM command line options lock. We can then load
  // it.
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

  return CompileResultWithMetadata{std::move(backend_result),
                                   std::move(compile_module_results)};
}

absl::StatusOr<std::unique_ptr<GpuExecutable>>
GpuLLVMCompiler::CompileToBackendResult(
    std::unique_ptr<HloModule> module, const GpuTopology& gpu_topology,
    const CompileOptions& options, se::StreamExecutor* stream_exec) {
  BinaryMap dnn_compiled_graphs;
  if (stream_exec) {
    se::dnn::DnnSupport* dnn_support = stream_exec->AsDnn();
    TF_RET_CHECK(dnn_support != nullptr);
    RETURN_IF_ERROR(RunCudnnCompilerPasses(module.get(), *dnn_support,
                                           &dnn_compiled_graphs));
  }

  llvm::LLVMContext llvm_context;
  TF_ASSIGN_OR_RETURN(
      CompileResultWithMetadata res,
      CompileToBackendResultImpl(module.get(), &llvm_context, gpu_topology,
                                 options, stream_exec));

  const DebugOptions& debug_opts = module->config().debug_options();
  bool embed_ir_in_executable = debug_opts.xla_embed_ir_in_executable();
  bool embed_debug_info = debug_opts.xla_gpu_executable_embed_debug_info();

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCreateGpuExecutable:#module=%s#",
                           module->name());
  });

  const se::DeviceDescription& gpu_device_info =
      gpu_topology.gpu_target_config().device_description;
  std::unique_ptr<GpuAliasInfo> alias_info = GetAliasInfo(gpu_device_info);

  TF_ASSIGN_OR_RETURN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      stream_executor::ExecutableAbiVersion::FromDeviceDescription(
          gpu_device_info));

  ModuleStats module_stats = res.backend_result.module_stats;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuExecutable> gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/embed_debug_info ? std::move(res.backend_result.asm_text)
                                        : std::string(),
          /*binary=*/std::move(res.backend_result.binary),
          /*dnn_compiled_graphs=*/std::move(dnn_compiled_graphs),
          /*executable=*/
          std::make_unique<ThunkExecutor>(
              std::move(res.compile_module_results.executable->thunks())),
          /*constants=*/std::move(res.compile_module_results.constants),
          /*globals=*/{},
          /*output_info=*/std::move(res.compile_module_results.output_info),
          /*module_name=*/std::move(res.compile_module_results.module_name),
          /*program_shape=*/
          module->compute_computation_layout().ComputeProgramShape(),
          /*mlir_allocations=*/
          (res.compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(res.compile_module_results.allocations)),
          /*buffer_assignment=*/
          std::move(res.compile_module_results.buffer_assignment),
          /*alias_info=*/std::move(alias_info),
          /*debug_options=*/debug_opts,
          /*device_description=*/gpu_device_info,
          /*debug_module=*/options.embed_hlo_module
              ? std::move(module)
              : std::unique_ptr<HloModule>(),
          /*enable_debug_info_manager=*/embed_debug_info,
          /*module_stats=*/std::move(module_stats),
          /*executable_abi_version=*/executable_abi_version,
          /*cpu_target_machine_options=*/
          options.cpu_target_config.has_value()
              ? options.cpu_target_config->cpu_target_machine_options
              : std::nullopt}));

  if (embed_ir_in_executable) {
    std::string ir_module_string_before_opt =
        llvm_ir::DumpToString(res.compile_module_results.llvm_modules[0].get());
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
    DCHECK_NE("", ir_module_string_before_opt);
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
  TF_ASSIGN_OR_RETURN(GpuTopology gpu_topology,
                      InferGpuTopology(hlo_module->config(), options.executor(),
                                       compile_options, options.debug_options(),
                                       PlatformId()));
  CHECK(gpu_topology.has_gpu_target_config());

  llvm::LLVMContext llvm_context;
  TF_ASSIGN_OR_RETURN(
      CompileResultWithMetadata res,
      CompileToBackendResultImpl(hlo_module.get(), &llvm_context, gpu_topology,
                                 compile_options, /*stream_exec=*/nullptr));

  std::vector<std::unique_ptr<CompiledModule>> results;
  ASSIGN_OR_RETURN(
      results.emplace_back(),
      LegacyGpuAotCompilationResult::FromModule(
          hlo_module.get(), res.compile_module_results.buffer_assignment.get(),
          res.backend_result.asm_text, res.backend_result.binary,
          res.backend_result.dnn_compiled_graphs, GetPointerSize(), this));

  return std::move(results);
}

absl::StatusOr<std::unique_ptr<CompiledModule>> GpuLLVMCompiler::Export(
    Executable* executable) {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
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
      &gpu_executable->module(), gpu_executable->buffer_assignment(),
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

  // Recreate HloModule+HloModuleConfig from proto.
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

  // Build the executable, which should be a thunk sequence.
  absl::string_view platform_name = PlatformId()->ToName();

  llvm::LLVMContext llvm_context;

  // Recreate BufferAssignment from proto.
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

  IrEmitterContext ir_emitter_context(
      hlo_module.get(), buffer_assignment.get(), &execution_stream_assignment,
      platform_name, device_description, mlir_context(), &llvm_context,
      /*emit_kernels=*/false, llvm::Triple(target_triple()), data_layout(),
      &kernel_compiler,
      cpu::TargetMachineOptions(hlo_module->config().debug_options()),
      &mlir_context_pool_);

  absl::string_view cache_file_path =
      hlo_module->config().debug_options().xla_gpu_kernel_cache_file();
  if (!cache_file_path.empty() &&
      hlo_module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism()) {
    RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }

  xla::llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock(
      GetLLVMCommandLineOptions(hlo_module->config().debug_options()));

  ThunkEmitter thunk_emitter(&ir_emitter_context, &llvm_options_lock);
  ASSIGN_OR_RETURN(auto sequential_thunk,
                   thunk_emitter.EmitHloEntryComputation(hlo_module.get()));

  // Get all other fields required by GpuExecutable.
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
  {
    tsl::profiler::TraceMe traceme("CreateGpuExecutable");
    std::unique_ptr<GpuAliasInfo> alias_info = GetAliasInfo(device_description);
    return GpuExecutable::Create(GpuExecutable::Params{
        /*asm_text=*/proto.asm_text(),
        /*binary=*/binary,
        /*dnn_compiled_graphs=*/
        BinaryMap(proto.dnn_compiled_graphs().cbegin(),
                  proto.dnn_compiled_graphs().cend()),
        /*executable=*/
        std::make_unique<ThunkExecutor>(std::move(sequential_thunk->thunks())),
        /*constants=*/std::move(constants),
        /*globals=*/{},
        /*output_info=*/std::move(output_info),
        /*module_name=*/std::move(hlo_module_name),
        /*program_shape=*/std::move(program_shape),
        /*mlir_allocations=*/std::nullopt,
        /*buffer_assignment=*/std::move(buffer_assignment),
        /*alias_info=*/std::move(alias_info),
        /*debug_options=*/std::move(debug_options),
        /*device_description=*/device_description,
        /*debug_module=*/std::move(hlo_module),
        /*enable_debug_info_manager=*/true,
        /*module_stats=*/{},
        /*executable_abi_version=*/executable_abi_version,
        /*cpu_target_machine_options=*/std::move(cpu_target_machine_options),
    });
  }
}

}  // namespace gpu
}  // namespace xla
