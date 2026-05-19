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

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/metal/metal_compute_capability.h"
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
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription(int ordinal) {
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

  auto desc = std::make_unique<DeviceDescription>();
  desc->set_name([[device name] UTF8String]);
  desc->set_device_vendor("Apple");

  const int generation = GetAppleFamilyGeneration(device);
  const bool metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
  desc->set_gpu_compute_capability(
      GpuComputeCapability(MetalComputeCapability(generation, metal3)));
  return desc;
}

absl::StatusOr<std::unique_ptr<Stream>> MetalExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> /*priority*/) {
  return absl::UnimplementedError("MetalStream is not implemented yet.");
}

absl::StatusOr<std::unique_ptr<Event>> MetalExecutor::CreateEvent() {
  return absl::UnimplementedError("MetalEvent is not implemented yet.");
}

DeviceAddressBase MetalExecutor::Allocate(uint64_t /*size*/,
                                          int64_t /*memory_space*/) {
  // No memory subsystem yet. Return a null DeviceAddressBase so callers see
  // the allocation as failed.
  return DeviceAddressBase();
}

void MetalExecutor::Deallocate(DeviceAddressBase* /*mem*/) {
  // No-op: nothing was allocated.
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MetalExecutor::HostMemoryAllocate(uint64_t /*size*/) {
  return absl::UnimplementedError(
      "MetalExecutor::HostMemoryAllocate is not implemented yet.");
}

bool MetalExecutor::SynchronizeAllActivity() {
  // Nothing to synchronize yet; report success.
  return true;
}

absl::Status MetalExecutor::SynchronousMemcpy(
    DeviceAddressBase* /*device_dst*/, const void* /*host_src*/,
    uint64_t /*size*/) {
  return absl::UnimplementedError(
      "MetalExecutor::SynchronousMemcpy (H2D) is not implemented yet.");
}

absl::Status MetalExecutor::SynchronousMemcpy(
    void* /*host_dst*/, const DeviceAddressBase& /*device_src*/,
    uint64_t /*size*/) {
  return absl::UnimplementedError(
      "MetalExecutor::SynchronousMemcpy (D2H) is not implemented yet.");
}

void MetalExecutor::DeallocateStream(Stream* /*stream*/) {
  // No-op: streams are not yet implemented.
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

}  // namespace metal
}  // namespace stream_executor
