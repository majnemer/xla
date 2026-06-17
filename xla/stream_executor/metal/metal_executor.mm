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

#include "xla/stream_executor/metal/metal_executor.h"

#include "xla/stream_executor/metal/metal_device_handle.h"

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/metal/metal_compute_capability.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/metal/metal_timer.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {
namespace {

// Returns the highest Apple<N> family generation the device supports, or 0
// if none of Apple7+ is supported (i.e., not an Apple Silicon GPU we target).
int GetAppleFamilyGeneration(id<MTLDevice> device) {
  // Probe newest-to-oldest. Apple1..Apple9 are the Apple Silicon GPU families.
  static constexpr struct {
    MTLGPUFamily family;
    int generation;
  } kFamilies[] = {
      {MTLGPUFamilyApple9, 9},  // M3 / A17
      {MTLGPUFamilyApple8, 8},  // M2 / A15-A16
      {MTLGPUFamilyApple7, 7},  // M1 / A14
  };
  for (const auto& f : kFamilies) {
    if ([device supportsFamily:f.family]) {
      return f.generation;
    }
  }
  return 0;
}

// Compatibility shim: kIOMainPortDefault was introduced in macOS 12 as the
// rename of the now-deprecated kIOMasterPortDefault. Both resolve to the
// default (NULL) mach port. Define it for SDKs that predate the rename.
#ifndef kIOMainPortDefault
#define kIOMainPortDefault kIOMasterPortDefault
#endif

// Apple GPU cores are 128-wide (128 FP32 ALUs per core) on every Apple Silicon
// family from Apple7 (M1 / A14) onward.
constexpr int kAppleAlusPerCore = 128;

// Queries the exact GPU core count from the IORegistry for `device`, returning
// 0 if it cannot be determined. Apple Silicon publishes an integer
// "gpu-core-count" property on the GPU's IOService entry; we locate that entry
// via the Metal device's registryID. This is the only reliable way to read the
// real core count, since Metal itself does not expose it. Requires linking the
// IOKit framework.
int QueryGpuCoreCountFromIOKit(id<MTLDevice> device) {
  const uint64_t registry_id = [device registryID];
  if (registry_id == 0) {
    return 0;
  }
  // IORegistryEntryIDMatching returns a dictionary with a +1 reference that is
  // consumed by IOServiceGetMatchingService, so we must not release it here.
  CFMutableDictionaryRef matching = IORegistryEntryIDMatching(registry_id);
  if (matching == nullptr) {
    return 0;
  }
  io_service_t service =
      IOServiceGetMatchingService(kIOMainPortDefault, matching);
  if (service == IO_OBJECT_NULL) {
    return 0;
  }
  int core_count = 0;
  // Search both directions: the property may sit on the matched entry, a
  // parent, or a child depending on the OS version's IOGPU topology.
  CFTypeRef value = IORegistryEntrySearchCFProperty(
      service, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
      kIORegistryIterateRecursively | kIORegistryIterateParents);
  if (value != nullptr) {
    if (CFGetTypeID(value) == CFNumberGetTypeID()) {
      CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType,
                       &core_count);
    }
    CFRelease(value);
  }
  IOObjectRelease(service);
  return core_count > 0 ? core_count : 0;
}

constexpr int64_t GBs(double gb_per_s) {
  return static_cast<int64_t>(gb_per_s * 1e9);
}
constexpr int64_t MiB(int64_t mib) { return mib * 1024 * 1024; }

// Approximate, *non-authoritative* per-SKU GPU characteristics for Apple
// Silicon. Apple does not expose memory bandwidth, GPU clock, or cache sizes
// through any public API, so these are representative published figures used
// only as estimates:
//   * memory_bandwidth is in bytes/s using decimal GB (how the figures are
//     normally quoted).
//   * clock_ghz is a nominal boost frequency (Apple publishes none).
//   * l2_cache_size is a coarse last-level / system-cache proxy.
// Values differ between binned and full-die variants of the same marketing
// name -- most notably the *_Max parts (e.g. M3 Max ships as 300 or 400 GB/s)
// -- so treat these as order-of-magnitude hints, not ground truth. core_count
// here is the full-die value and a fallback only; QueryGpuCoreCountFromIOKit is
// preferred whenever it succeeds.
struct AppleGpuSpec {
  const char* name_substr;
  int core_count;
  double clock_ghz;
  int64_t memory_bandwidth;
  int64_t l2_cache_size;
};

// Ordered most-specific first within each generation so substring matching
// resolves "M3 Max" before the bare "M3", etc.
const AppleGpuSpec kAppleGpuSpecs[] = {
    // M4 family (figures provisional).
    {"M4 Max", 40, 1.40, GBs(546), MiB(32)},
    {"M4 Pro", 20, 1.40, GBs(273), MiB(24)},
    {"M4", 10, 1.40, GBs(120), MiB(8)},
    // M3 family.
    {"M3 Ultra", 80, 1.40, GBs(819), MiB(48)},
    {"M3 Max", 40, 1.40, GBs(400), MiB(32)},
    {"M3 Pro", 18, 1.40, GBs(150), MiB(12)},
    {"M3", 10, 1.40, GBs(100), MiB(8)},
    // M2 family.
    {"M2 Ultra", 76, 1.40, GBs(800), MiB(48)},
    {"M2 Max", 38, 1.40, GBs(400), MiB(32)},
    {"M2 Pro", 19, 1.40, GBs(200), MiB(24)},
    {"M2", 10, 1.40, GBs(100), MiB(8)},
    // M1 family.
    {"M1 Ultra", 64, 1.296, GBs(800), MiB(48)},
    {"M1 Max", 32, 1.296, GBs(400), MiB(32)},
    {"M1 Pro", 16, 1.296, GBs(200), MiB(24)},
    {"M1", 8, 1.278, GBs(68.25), MiB(8)},
};

const AppleGpuSpec* FindAppleGpuSpec(const std::string& device_name) {
  for (const AppleGpuSpec& spec : kAppleGpuSpecs) {
    if (device_name.find(spec.name_substr) != std::string::npos) {
      return &spec;
    }
  }
  return nullptr;
}

// Conservative per-generation fallbacks used when the device name is not in the
// table above (e.g. an A-series GPU, or a future part).
double FallbackClockGhz(int generation) {
  if (generation >= 9) return 1.40;
  if (generation == 8) return 1.398;
  if (generation == 7) return 1.278;
  return 1.0;
}

int64_t FallbackBandwidth(int generation) {
  if (generation >= 8) return GBs(100);
  if (generation == 7) return GBs(68.25);
  return GBs(50);
}

MemorySpace NormalizeMetalAllocationMemorySpace(MemorySpace type) {
  // Metal has no distinct collective memory implementation; collective
  // allocations use ordinary device-backed MTLBuffers.
  return type == MemorySpace::kCollective ? MemorySpace::kDevice : type;
}

}  // namespace

MetalExecutor::MetalExecutor(Platform* platform, int ordinal)
    : gpu::GpuExecutor(platform, ordinal), device_(nil) {}

MetalExecutor::~MetalExecutor() = default;

fft::FftSupport* MetalExecutor::AsFft() {
  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::FftFactory> factory =
      registry->GetFactory<PluginRegistry::FftFactory>(kMetalPlatformId);
  if (!factory.ok()) {
    LOG(ERROR) << "Unable to retrieve Metal FFT factory: "
               << factory.status().message();
    return nullptr;
  }
  fft_.reset(factory.value()(this));
  return fft_.get();
}

absl::Status MetalExecutor::Init() {
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  const int ordinal = device_ordinal();
  if (devices == nil ||
      ordinal < 0 ||
      ordinal >= static_cast<int>([devices count])) {
    return absl::NotFoundError(
        absl::StrCat("Metal device ordinal ", ordinal, " is out of range; ",
                     "system reports ",
                     devices == nil ? 0 : static_cast<int>([devices count]),
                     " device(s)."));
  }
  device_ = devices[ordinal];
  allocator_ = std::make_unique<MetalAllocator>(device_);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription(int ordinal) {
  std::unique_ptr<DeviceDescription> desc;
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices == nil ||
        ordinal < 0 ||
        ordinal >= static_cast<int>([devices count])) {
      return absl::NotFoundError(
          absl::StrCat("Metal device ordinal ", ordinal, " is out of range; ",
                       "system reports ",
                       devices == nil ? 0 : static_cast<int>([devices count]),
                       " device(s)."));
    }
    id<MTLDevice> device = devices[ordinal];

    desc = std::make_unique<DeviceDescription>();
    const char* raw_name = [[device name] UTF8String];
    const std::string device_name = raw_name != nullptr ? raw_name : "";
    desc->set_name(device_name);
    desc->set_device_vendor("Apple");

    // Apple Silicon shares system memory; recommendedMaxWorkingSetSize is
    // the closest "device memory" scalar. GpuCompiler's topology inference
    // rejects the default -1.
    desc->set_device_memory_size(
        static_cast<int64_t>([device recommendedMaxWorkingSetSize]));

    // Apple Silicon is a 64-bit architecture with a unified host/device address
    // space. Not exposed by Metal, but invariant for every supported part.
    desc->set_device_address_bits(64);

    const int generation = GetAppleFamilyGeneration(device);
    const bool metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
    desc->set_gpu_compute_capability(
        GpuComputeCapability(MetalComputeCapability(generation, metal3)));

    // Populate launch-dim fields the GPU pipeline reads; defaults are
    // kUninitialized (-1) which underflows. Apple's per-axis threadgroup max
    // is an unordered triple, so min() defensively. SIMD width 32 holds for
    // every supported Apple Silicon family. No per-axis grid-size limit
    // exposed; permissive default keeps launch math from tripping.
    const MTLSize max_threadgroup = [device maxThreadsPerThreadgroup];
    const NSUInteger max_threadgroup_total = std::min(
        {max_threadgroup.width, max_threadgroup.height, max_threadgroup.depth});
    desc->set_threads_per_block_limit(
        static_cast<int64_t>(max_threadgroup_total));
    desc->set_threads_per_warp(
        std::min<int64_t>(32, static_cast<int64_t>(max_threadgroup_total)));
    desc->set_block_dim_limit(BlockDim(
        /*x=*/std::numeric_limits<int64_t>::max(),
        /*y=*/std::numeric_limits<int64_t>::max(),
        /*z=*/std::numeric_limits<int64_t>::max()));

    // --- Shared (threadgroup) memory -------------------------------------
    // Metal exposes the per-threadgroup limit directly via
    // maxThreadgroupMemoryLength. Apple GPUs do not distinguish a separate
    // higher "opt-in" tier the way CUDA does, and the physical per-core
    // threadgroup memory equals this per-block maximum, so all three
    // shared-memory scalars take the same queried value. (Smaller blocks can
    // still co-reside on a core, since occupancy modeling divides
    // shared_memory_per_core by the per-block usage.)
    int64_t shared_memory_per_block =
        static_cast<int64_t>([device maxThreadgroupMemoryLength]);
    if (shared_memory_per_block <= 0) {
      shared_memory_per_block = 32 * 1024;  // 32 KiB: the Apple Silicon norm.
    }
    desc->set_shared_memory_per_block(shared_memory_per_block);
    desc->set_shared_memory_per_block_optin(shared_memory_per_block);
    desc->set_shared_memory_per_core(shared_memory_per_block);

    // --- Execution units per core ----------------------------------------
    // 128 FP32 ALUs per GPU core across all Apple Silicon families.
    desc->set_fpus_per_core(kAppleAlusPerCore);

    // --- Core count (query first, then estimate) -------------------------
    const AppleGpuSpec* spec = FindAppleGpuSpec(device_name);
    int core_count = QueryGpuCoreCountFromIOKit(device);
    if (core_count <= 0) {
      // IOKit lookup failed: fall back to the per-SKU table.
      core_count = spec != nullptr ? spec->core_count : 0;
    }
    if (core_count <= 0) {
      // Last resort so topology inference never divides by zero. 8 is the
      // smallest shipping Apple Silicon GPU configuration.
      core_count = 8;
    }
    desc->set_core_count(core_count);

    // --- Clock, bandwidth, cache (estimated; not exposed by Metal) -------
    // None of these are available through any public Metal API, so they are
    // estimated from the device name where known and from the GPU family
    // generation otherwise. See AppleGpuSpec for the caveats.
    desc->set_clock_rate_ghz(static_cast<float>(
        spec != nullptr ? spec->clock_ghz : FallbackClockGhz(generation)));
    desc->set_memory_bandwidth(spec != nullptr ? spec->memory_bandwidth
                                               : FallbackBandwidth(generation));
    desc->set_l2_cache_size(spec != nullptr ? spec->l2_cache_size : MiB(8));
  }
  return desc;
}

absl::StatusOr<std::unique_ptr<Stream>> MetalExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MetalStream> stream,
                      MetalStream::Create(this, priority));
  {
    absl::MutexLock lock(&streams_mu_);
    streams_[stream.get()] = 1;
  }
  return stream;
}

absl::StatusOr<std::unique_ptr<Event>> MetalExecutor::CreateEvent() {
  return MetalEvent::Create(this);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
MetalExecutor::CreateEventBasedTimer(Stream* stream,
                                     bool /*use_delay_kernel*/) {
  auto* metal_stream = dynamic_cast<MetalStream*>(stream);
  if (metal_stream == nullptr) {
    return absl::InvalidArgumentError(
        "MetalExecutor::CreateEventBasedTimer: stream is not a MetalStream.");
  }
  TF_ASSIGN_OR_RETURN(MetalTimer timer, MetalTimer::Create(this, metal_stream));
  return std::make_unique<MetalTimer>(std::move(timer));
}

absl::StatusOr<id<MTLLibrary>> MetalExecutor::LoadLibraryFromMsl(
    const char* source) {
  auto it = source_to_library_.find(source);
  if (it != source_to_library_.end()) {
    ++it->second.second;
    return it->second.first;
  }
  id<MTLDevice> device = device_;
  if (device == nil) {
    return absl::FailedPreconditionError(
        "MetalExecutor::LoadLibraryFromMsl: device is nil; was Init() "
        "called?");
  }
  // MSL compilation autoreleases internal temporaries; bound them here.
  id<MTLLibrary> library;
  @autoreleasepool {
    NSString* source_ns = [[NSString alloc] initWithUTF8String:source];
    if (source_ns == nil) {
      return absl::InvalidArgumentError(
          "MetalExecutor::LoadLibraryFromMsl: MSL source is not valid UTF-8.");
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    library =
        [device newLibraryWithSource:source_ns options:options error:&error];
    if (library == nil) {
      NSString* msg = error == nil ? @"(no error info)" : [error description];
      return absl::InternalError(
          absl::StrCat("MetalExecutor::LoadLibraryFromMsl: "
                       "newLibraryWithSource failed: ",
                       [msg UTF8String]));
    }
    source_to_library_.emplace(source, std::make_pair(library, uint64_t{1}));
  }
  return library;
}

void MetalExecutor::UnloadKernel(const Kernel* kernel) {
  absl::MutexLock lock(in_memory_libraries_mu_);
  auto src_it = kernel_to_source_.find(kernel);
  if (src_it == kernel_to_source_.end()) {
    return;  // Never recorded — nothing to release.
  }
  const char* source = src_it->second;
  kernel_to_source_.erase(src_it);
  auto lib_it = source_to_library_.find(source);
  if (lib_it == source_to_library_.end()) {
    return;
  }
  if (--lib_it->second.second == 0) {
    source_to_library_.erase(lib_it);
  }
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  auto msl = spec.msl_source_in_memory();
  if (!msl.has_value()) {
    return absl::InvalidArgumentError(
        "MetalExecutor::LoadKernel: spec does not contain MSL source — "
        "Metal only supports MslSourceInMemory / OwningMslSourceInMemory.");
  }
  const char* source_ptr = msl->source.data();
  id<MTLLibrary> library = nil;
  {
    absl::MutexLock lock(in_memory_libraries_mu_);
    auto cached = LoadLibraryFromMsl(source_ptr);
    if (!cached.ok()) {
      return cached.status();
    }
    library = *cached;
  }
  auto kernel = MetalKernel::Create(this, library, spec.kernel_name(),
                                    static_cast<unsigned>(spec.arity()));
  if (!kernel.ok()) {
    return kernel.status();
  }
  (*kernel)->set_name(spec.kernel_name());
  {
    absl::MutexLock lock(in_memory_libraries_mu_);
    kernel_to_source_.emplace(kernel->get(), source_ptr);
  }
  const auto& packing = spec.kernel_args_packing();
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          packing)) {
    (*kernel)->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(packing));
  } else {
    const auto& packing_spec = std::get<KernelArgsPackingSpec>(packing);
    (*kernel)->set_args_packing(
        [packing_spec](const Kernel& /*kernel*/, const KernelArgs& args) {
          const auto& mem_args = Cast<KernelArgsDeviceAddressArray>(&args);
          return packing_spec.BuildArguments(mem_args->packed_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return std::move(*kernel);
}

DeviceAddressBase MetalExecutor::Allocate(uint64_t size,
                                          int64_t memory_space) {
  if (allocator_ == nullptr) {
    return DeviceAddressBase();
  }
  MemorySpace type = NormalizeMetalAllocationMemorySpace(
      static_cast<MemorySpace>(memory_space));
  auto base = allocator_->Allocate(size, type);
  if (!base.ok()) {
    return DeviceAddressBase();
  }
  return DeviceAddressBase(*base, size);
}

void MetalExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || allocator_ == nullptr) {
    return;
  }
  allocator_->Deallocate(mem->opaque());
  *mem = DeviceAddressBase();
}

absl::StatusOr<ModuleHandle> MetalExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  auto module = std::make_unique<MetalModule>();
  for (const MultiModuleLoaderSpec::ConstantSpec& constant : spec.constants()) {
    const uint64_t size = static_cast<uint64_t>(constant.initial_bytes.size());
    DeviceAddressBase addr = Allocate(size, /*memory_space=*/0);
    if (addr.opaque() == nullptr && size > 0) {
      for (auto& [_, a] : module->symbols) {
        Deallocate(&a);
      }
      return absl::ResourceExhaustedError(absl::StrCat(
          "Metal constant allocation of ", size, " bytes failed for symbol ",
          constant.symbol_name));
    }
    module->symbols.emplace(std::string(constant.symbol_name), addr);
  }
  ModuleHandle handle(module.get());
  absl::MutexLock lock(&modules_mu_);
  modules_.emplace(handle, std::move(module));
  return handle;
}

bool MetalExecutor::UnloadModule(ModuleHandle module_handle) {
  std::unique_ptr<MetalModule> module;
  {
    absl::MutexLock lock(&modules_mu_);
    auto it = modules_.find(module_handle);
    if (it == modules_.end()) {
      return false;
    }
    module = std::move(it->second);
    modules_.erase(it);
  }
  for (auto& [_, addr] : module->symbols) {
    Deallocate(&addr);
  }
  return true;
}

absl::StatusOr<DeviceAddressBase> MetalExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  absl::MutexLock lock(&modules_mu_);
  auto it = modules_.find(module_handle);
  if (it == modules_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Metal module not loaded for symbol lookup: ", symbol_name));
  }
  auto sit = it->second->symbols.find(symbol_name);
  if (sit == it->second->symbols.end()) {
    return absl::NotFoundError(
        absl::StrCat("Metal symbol not found: ", symbol_name));
  }
  return sit->second;
}

namespace {

// MemoryAllocation backed by an allocator-owned Shared MTLBuffer. On Apple
// Silicon (unified memory) the buffer's contents pointer is both the host
// address and a value Stream::Memcpy can Resolve back to (MTLBuffer, offset),
// so device and host allocations are the same physical class.
class MetalHostMemoryAllocation : public MemoryAllocation {
 public:
  MetalHostMemoryAllocation(MetalAllocator* allocator, void* base,
                            uint64_t size)
      : allocator_(allocator), base_(base), size_(size) {}

  ~MetalHostMemoryAllocation() override {
    if (base_ != nullptr) {
      allocator_->Deallocate(base_);
    }
  }

  DeviceAddressBase address() const override {
    return DeviceAddressBase(base_, size_);
  }

 private:
  MetalAllocator* allocator_;
  void* base_;
  uint64_t size_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MetalExecutor::HostMemoryAllocate(uint64_t size) {
  if (allocator_ == nullptr) {
    return absl::FailedPreconditionError(
        "MetalExecutor::HostMemoryAllocate: executor not initialized.");
  }
  auto base = allocator_->Allocate(size, MemorySpace::kHost);
  if (!base.ok()) {
    return base.status();
  }
  return std::make_unique<MetalHostMemoryAllocation>(allocator_.get(), *base,
                                                    size);
}

absl::StatusOr<MemorySpace> MetalExecutor::GetPointerMemorySpace(
    const void* ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError(
        "MetalExecutor::GetPointerMemorySpace: null pointer.");
  }
  if (allocator_ == nullptr) {
    return absl::FailedPreconditionError(
        "MetalExecutor::GetPointerMemorySpace: executor not initialized.");
  }
  if (auto resolved = allocator_->Resolve(ptr); resolved.has_value()) {
    return resolved->memory_space;
  }
  return absl::NotFoundError(
      "MetalExecutor::GetPointerMemorySpace: pointer is not owned by the "
      "Metal allocator.");
}

bool MetalExecutor::SynchronizeAllActivity() {
  // Mirror cuCtxSynchronize: block until every stream we created is idle.
  // Snapshot under the lock and mark each stream as draining so a concurrent
  // DeallocateStream can't free one while we still hold its pointer. We must
  // not hold streams_mu_ across BlockHostUntilDone — a host callback may
  // create or destroy a stream (taking streams_mu_), which would deadlock.
  std::vector<Stream*> streams;
  {
    absl::MutexLock lock(&streams_mu_);
    streams.reserve(streams_.size());
    for (auto& [s, count] : streams_) {
      streams.push_back(s);
      ++count;
    }
  }
  bool ok = true;
  for (Stream* s : streams) {
    if (!s->BlockHostUntilDone().ok()) {
      ok = false;
    }
  }
  {
    absl::MutexLock lock(&streams_mu_);
    for (Stream* s : streams) {
      auto it = streams_.find(s);
      CHECK(it != streams_.end());
      --it->second;
    }
  }
  return ok;
}

absl::Status MetalExecutor::SynchronousMemcpy(
    DeviceAddressBase* device_dst, const void* host_src, uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  if (device_dst == nullptr || device_dst->opaque() == nullptr ||
      host_src == nullptr) {
    return absl::InvalidArgumentError(
        "MetalExecutor::SynchronousMemcpy (H2D): null pointer.");
  }
  // CUDA's cuMemcpyHtoD is synchronous w.r.t. previously queued GPU work on
  // the device. Apple Silicon's shared MTLBuffer makes the device pointer
  // host-addressable, but a plain CPU memcpy would race against an in-flight
  // kernel touching this region. Drain every stream first, then memcpy.
  if (!SynchronizeAllActivity()) {
    return absl::InternalError(
        "MetalExecutor::SynchronousMemcpy (H2D): a stream returned an error "
        "while draining prior work.");
  }
  std::memcpy(device_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status MetalExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  if (host_dst == nullptr || device_src.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        "MetalExecutor::SynchronousMemcpy (D2H): null pointer.");
  }
  // Drain pending GPU work that may still be writing to the source region.
  if (!SynchronizeAllActivity()) {
    return absl::InternalError(
        "MetalExecutor::SynchronousMemcpy (D2H): a stream returned an error "
        "while draining prior work.");
  }
  std::memcpy(host_dst, device_src.opaque(), size);
  return absl::OkStatus();
}

void MetalExecutor::DeallocateStream(Stream* stream) {
  if (stream == nullptr) return;
  absl::MutexLock lock(&streams_mu_);
  // Block until any in-flight SynchronizeAllActivity has released its
  // references — otherwise its BlockHostUntilDone call could outlive the
  // object. The stream's own liveness holds the count at 1; a count above
  // that means a drain still references it. Erasing drops the liveness.
  const auto drained = [this, stream]()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(streams_mu_) {
        auto it = streams_.find(stream);
        CHECK(it != streams_.end());
        return it->second == 1;
      };
  streams_mu_.Await(absl::Condition(&drained));
  streams_.erase(stream);
}

absl::Status MetalExecutor::EnablePeerAccessTo(StreamExecutor* /*other*/) {
  return absl::UnimplementedError(
      "Metal does not support peer access between devices.");
}

bool MetalExecutor::CanEnablePeerAccessTo(StreamExecutor* /*other*/) {
  return false;
}

int64_t MetalExecutor::GetMemoryLimitBytes() const {
  if (device_ == nil) {
    return 0;
  }
  return static_cast<int64_t>([device_ recommendedMaxWorkingSetSize]);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MetalExecutor::CreateMemoryAllocator(MemorySpace type) {
  if (allocator_ == nullptr) {
    return absl::FailedPreconditionError(
        "MetalExecutor::CreateMemoryAllocator: executor not initialized.");
  }
  // Unified memory: every memory space backs onto the same Shared-mode
  // MTLBuffer pool.
  if (type == MemorySpace::kDevice || type == MemorySpace::kUnified ||
      type == MemorySpace::kCollective || type == MemorySpace::kHost) {
    return std::make_unique<GenericMemoryAllocator>(
        [this, type](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          MemorySpace allocation_type =
              NormalizeMetalAllocationMemorySpace(type);
          TF_ASSIGN_OR_RETURN(void* base,
                              allocator_->Allocate(size, allocation_type));
          return std::make_unique<GenericMemoryAllocation>(
              base, size, [this](void* ptr, uint64_t /*size*/) {
                allocator_->Deallocate(ptr);
              });
        });
  }
  return absl::UnimplementedError(absl::StrCat(
      "MetalExecutor::CreateMemoryAllocator: memory space ",
      static_cast<int>(type), " is not supported."));
}

bool MetalExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  if (device_ == nil) {
    return false;
  }
  const int64_t total_bytes =
      static_cast<int64_t>([device_ recommendedMaxWorkingSetSize]);
  const int64_t allocated =
      static_cast<int64_t>([device_ currentAllocatedSize]);
  *total = total_bytes;
  *free = std::max<int64_t>(0, total_bytes - allocated);
  return true;
}

void* GetMetalDeviceOpaque(StreamExecutor* stream_exec) {
  if (stream_exec == nullptr) return nullptr;
  auto* metal_executor = dynamic_cast<MetalExecutor*>(stream_exec);
  if (metal_executor == nullptr) return nullptr;
  id<MTLDevice> device = metal_executor->device();
  if (device == nil) return nullptr;
  return (__bridge void*)device;
}

}  // namespace metal
}  // namespace stream_executor
