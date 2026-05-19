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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace metal {

// Per-device StreamExecutor for the Metal platform.
//
// Inherits from gpu::GpuExecutor, the shared base for GPU-class executors. It
// supplies device_ordinal_, the API-trace logger plumbing, and a multicast
// hook (defaults to unsupported) — all GPU-generic, none CUDA-specific. The
// Activate() no-op behavior comes from StreamExecutorCommon and is correct
// for Metal, which has no thread-local context analogous to a CUDA context.
//
// This header is Obj-C++ — it imports <Metal/Metal.h> and exposes id<MTLDevice>
// in the private section. Per the boundary discipline in
// xla/stream_executor/metal/, headers in this directory may freely use Metal
// types; consumers in this directory are .mm files. The rest of XLA interacts
// through the abstract StreamExecutor base.
//
// v1 status: most operations return absl::UnimplementedError. Init() acquires
// the MTLDevice and validates the ordinal; CreateDeviceDescription populates
// device name + Apple Silicon family info. The rest fills in as MetalStream /
// MetalEvent / MetalKernel / memory allocation land.
class MetalExecutor : public gpu::GpuExecutor {
 public:
  MetalExecutor(Platform* platform, int ordinal);
  ~MetalExecutor() override;

  // StreamExecutor interface — overrides not handled by GpuExecutor.
  absl::Status Init() override;

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;

  // Mirrors the CUDA driver's cuModuleLoad/cuModuleUnload pairing: a single
  // call allocates device memory for every constant in `spec`, and the matching
  // UnloadModule frees them. There is no MSL involved here — MSL libraries are
  // built on demand by LoadKernel — so `spec`'s cubin/PTX fields are ignored.
  // The returned handle keys an internal symbol table consulted by GetSymbol.
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  absl::StatusOr<DeviceAddressBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return MetalExecutor::CreateDeviceDescription(device_ordinal());
  }

  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace memory_space) override;

  bool SynchronizeAllActivity() override;
  absl::Status SynchronousMemcpy(DeviceAddressBase* device_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) override;

  void DeallocateStream(Stream* stream) override;
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;

  // Memory limit reported by recommendedMaxWorkingSetSize on the MTLDevice.
  // Overrides the env-driven default in StreamExecutorCommon.
  int64_t GetMemoryLimitBytes() const override;

  // Reports total = MTLDevice.recommendedMaxWorkingSetSize (matches what
  // we already publish as GetMemoryLimitBytes and device_memory_size on
  // the DeviceDescription); free = total - currentAllocatedSize, the
  // closest analogue to the CUDA/ROCm contract on Apple's unified-memory
  // model. Consumed by GPU PJRT's BFC allocator sizing.
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  // Static helper used by both the virtual override above and by
  // MetalPlatform::DescriptionForDevice (which is const and therefore can't
  // construct a transient executor).
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int ordinal);

  // Returns the underlying MTLDevice. Available only inside .mm consumers.
  id<MTLDevice> device() const { return device_; }

  // Per-executor allocator. Created in Init(). Used internally by Allocate/
  // Deallocate; exposed for blit encoders and kernel binding code that need
  // to resolve a DeviceAddressBase byte pointer back to (MTLBuffer, offset).
  MetalAllocator* allocator() { return allocator_.get(); }

 private:
  // JIT-compile (or fetch the cached result of) `[device newLibraryWithSource:]`
  // for the MSL pointed at by `source`. Keying by source pointer mirrors
  // CudaExecutor::LoadModuleFromCuBin, which keys by cubin pointer: same
  // GpuExecutable instance always points at the same backing buffer, so
  // repeat LoadKernel calls coalesce. Refcount semantics mirror the CUDA
  // path for consistency, though MTLLibrary lifetime is otherwise managed
  // by ARC.
  absl::StatusOr<id<MTLLibrary>> LoadLibraryFromMsl(const char* source)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_libraries_mu_);

  // The Metal device handle. Populated by Init() and immutable thereafter.
  id<MTLDevice> device_;

  // Per-device allocator. Created in Init() once `device_` is set.
  std::unique_ptr<MetalAllocator> allocator_;

  // Guards the cached MTLLibrary table.
  absl::Mutex in_memory_libraries_mu_;

  // Source pointer -> {MTLLibrary, reference count}. Mirror of
  // CudaExecutor::gpu_binary_to_module_.
  absl::flat_hash_map<const char*, std::pair<id<MTLLibrary>, uint64_t>>
      source_to_library_ ABSL_GUARDED_BY(in_memory_libraries_mu_);

  // Kernel -> the source pointer it was loaded from, for unload bookkeeping.
  // Mirror of CudaExecutor::kernel_to_gpu_binary_.
  absl::flat_hash_map<const Kernel*, const char*> kernel_to_source_
      ABSL_GUARDED_BY(in_memory_libraries_mu_);

  // Streams created by CreateStream() and not yet destroyed, mapped to a
  // holder refcount: 1 for the stream's own liveness plus 1 per in-flight
  // SynchronizeAllActivity draining it. The drain releases streams_mu_ before
  // calling BlockHostUntilDone (a host callback may create/destroy a stream
  // and would otherwise deadlock on the lock), so DeallocateStream drops the
  // liveness count and waits for the refcount to reach 0 — i.e. no drain still
  // holds the raw pointer — before letting the object be destroyed.
  absl::Mutex streams_mu_;
  absl::flat_hash_map<Stream*, int> streams_ ABSL_GUARDED_BY(streams_mu_);

  // Backing storage for a single LoadModule call. The ModuleHandle returned
  // by LoadModule uses the address of one of these objects as its opaque id,
  // so handles remain unique for the executor's lifetime.
  struct MetalModule {
    absl::flat_hash_map<std::string, DeviceAddressBase> symbols;
  };

  absl::Mutex modules_mu_;
  absl::flat_hash_map<ModuleHandle, std::unique_ptr<MetalModule>> modules_
      ABSL_GUARDED_BY(modules_mu_);

  MetalExecutor(const MetalExecutor&) = delete;
  MetalExecutor& operator=(const MetalExecutor&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
