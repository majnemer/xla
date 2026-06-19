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

#include "xla/stream_executor/metal/metal_platform.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {
namespace metal {

MetalPlatform::MetalPlatform() : name_(kMetalPlatformId->ToName()) {}

PlatformId MetalPlatform::id() const { return kMetalPlatformId; }

const std::string &MetalPlatform::Name() const { return name_; }

int MetalPlatform::VisibleDeviceCount() const {
  @autoreleasepool {
    NSArray *devices = MTLCopyAllDevices();
    if (devices == nil) {
      return 0;
    }
    return static_cast<int>([devices count]);
  }
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalPlatform::DescriptionForDevice(int ordinal) const {
  return MetalExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor *> MetalPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal,
      [this, ordinal]() -> absl::StatusOr<std::unique_ptr<StreamExecutor>> {
        auto executor = std::make_unique<MetalExecutor>(this, ordinal);
        TF_RETURN_IF_ERROR(executor->Init());
        return executor;
      });
}

absl::StatusOr<StreamExecutor *> MetalPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

} // namespace metal

static void InitializeMetalPlatform() {
  auto status = PlatformManager::PlatformWithName("METAL");
  if (!status.ok()) {
    CHECK_OK(PlatformManager::RegisterPlatform(
        std::make_unique<metal::MetalPlatform>()));
  }
}

} // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    metal_platform, stream_executor::InitializeMetalPlatform());
