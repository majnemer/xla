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

#include "xla/stream_executor/metal/metal_event.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/metal/metal_executor.h"

namespace stream_executor {
namespace metal {

struct MetalEventErrorState {
  absl::Mutex mu;
  // Error for the currently relevant signal value. A reused event can publish
  // a later value; old errors are not treated as errors for the later record.
  uint64_t error_value = 0;
  absl::Status error_status = absl::OkStatus();
};

MetalEvent::MetalEvent(id<MTLSharedEvent> shared_event)
    : shared_event_(shared_event),
      error_state_(std::make_shared<MetalEventErrorState>()) {}

MetalEvent::~MetalEvent() = default;

absl::StatusOr<std::unique_ptr<MetalEvent>> MetalEvent::Create(
    MetalExecutor* executor) {
  if (executor == nullptr) {
    return absl::InvalidArgumentError(
        "MetalEvent::Create: executor is null.");
  }
  id<MTLDevice> device = executor->device();
  if (device == nil) {
    return absl::FailedPreconditionError(
        "MetalEvent::Create: executor has no MTLDevice; was Init() called?");
  }
  id<MTLSharedEvent> shared_event = [device newSharedEvent];
  if (shared_event == nil) {
    return absl::ResourceExhaustedError(
        "MetalEvent::Create: [MTLDevice newSharedEvent] returned nil.");
  }
  // 0 means "never recorded" in this backend.
  shared_event.signaledValue = 0;
  return std::unique_ptr<MetalEvent>(new MetalEvent(shared_event));
}

void MetalEvent::MarkSignalErrorForValue(
    const std::shared_ptr<MetalEventErrorState>& state, uint64_t value,
    absl::Status status) {
  if (state == nullptr || status.ok()) return;
  absl::MutexLock lock(&state->mu);
  // Newer record always wins: a fresher cmd_buf failing supersedes whatever
  // we had recorded for an earlier value.
  if (value > state->error_value) {
    state->error_value = value;
    state->error_status = std::move(status);
    return;
  }
  // Same-value race: first non-OK status wins.
  if (value == state->error_value && state->error_status.ok()) {
    state->error_status = std::move(status);
    return;
  }
  // Older value than what we already have an error for; ignore.
}

void MetalEvent::MarkSignalErrorForValue(uint64_t value, absl::Status status) {
  MarkSignalErrorForValue(error_state_, value, std::move(status));
}

std::optional<absl::Status> MetalEvent::ErrorForValue(uint64_t value) const {
  absl::MutexLock lock(&error_state_->mu);
  if (error_state_->error_value == value &&
      !error_state_->error_status.ok()) {
    return error_state_->error_status;
  }
  return std::nullopt;
}

Event::Status MetalEvent::PollForStatus() {
  const uint64_t value = recorded_value();
  // Unrecorded event represents a point before any work — already satisfied.
  if (value == 0) {
    return Event::Status::kComplete;
  }
  if (ErrorForValue(value).has_value()) {
    return Event::Status::kError;
  }
  if (shared_event_.signaledValue >= value) {
    return Event::Status::kComplete;
  }
  // Re-check after reading signaledValue to avoid returning pending if the
  // signal command buffer failed concurrently.
  if (ErrorForValue(value).has_value()) {
    return Event::Status::kError;
  }
  return Event::Status::kPending;
}

absl::Status MetalEvent::Synchronize() {
  const uint64_t value = recorded_value();
  if (value == 0) {
    return absl::OkStatus();
  }
  // Do not use one unbounded wait. If the cmd_buf that should signal the
  // value errors before signaling, the shared event may never reach `value`.
  // Poll in bounded chunks so cmd_buf completion handlers can surface the
  // event error.
  constexpr uint64_t kWaitChunkMs = 100;
  for (;;) {
    if (std::optional<absl::Status> error = ErrorForValue(value);
        error.has_value()) {
      return *std::move(error);
    }
    if (shared_event_.signaledValue >= value) {
      return absl::OkStatus();
    }
    @autoreleasepool {
      BOOL signaled = [shared_event_ waitUntilSignaledValue:value
                                                  timeoutMS:kWaitChunkMs];
      if (signaled) {
        return absl::OkStatus();
      }
    }
  }
}

}  // namespace metal
}  // namespace stream_executor
