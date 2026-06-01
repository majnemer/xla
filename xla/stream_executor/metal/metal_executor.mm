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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

}  // namespace

MetalExecutor::MetalExecutor(Platform* platform, int ordinal)
    : gpu::GpuExecutor(platform, ordinal), device_(nil) {}

MetalExecutor::~MetalExecutor() = default;

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
    desc->set_name([[device name] UTF8String]);
    desc->set_device_vendor("Apple");

    // Apple Silicon shares system memory; recommendedMaxWorkingSetSize is
    // the closest "device memory" scalar. GpuCompiler's topology inference
    // rejects the default -1.
    desc->set_device_memory_size(
        static_cast<int64_t>([device recommendedMaxWorkingSetSize]));

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
          return packing_spec.BuildArguments(mem_args->device_addr_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return std::move(*kernel);
}

DeviceAddressBase MetalExecutor::Allocate(uint64_t size,
                                          int64_t /*memory_space*/) {
  if (allocator_ == nullptr) {
    return DeviceAddressBase();
  }
  auto base = allocator_->Allocate(size);
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
  auto base = allocator_->Allocate(size);
  if (!base.ok()) {
    return base.status();
  }
  return std::make_unique<MetalHostMemoryAllocation>(allocator_.get(), *base,
                                                    size);
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
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          TF_ASSIGN_OR_RETURN(void* base, allocator_->Allocate(size));
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

}  // namespace metal
}  // namespace stream_executor
