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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/memory_allocation.h"
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

  // Static helper used by both the virtual override above and by
  // MetalPlatform::DescriptionForDevice (which is const and therefore can't
  // construct a transient executor).
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int ordinal);

  // Returns the underlying MTLDevice. Available only inside .mm consumers.
  id<MTLDevice> device() const { return device_; }

 private:
  // The Metal device handle. Populated by Init() and immutable thereafter.
  id<MTLDevice> device_;

  MetalExecutor(const MetalExecutor&) = delete;
  MetalExecutor& operator=(const MetalExecutor&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
