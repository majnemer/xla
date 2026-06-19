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
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
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

MetalEvent::MetalEvent(id<MTLSharedEvent> shared_event, bool allow_timing)
    : shared_event_(shared_event), allow_timing_(allow_timing),
      error_state_(std::make_shared<MetalEventErrorState>()) {}

MetalEvent::~MetalEvent() = default;

absl::StatusOr<std::unique_ptr<MetalEvent>>
MetalEvent::Create(MetalExecutor *executor, bool allow_timing) {
  @autoreleasepool {
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
    return std::unique_ptr<MetalEvent>(
        new MetalEvent(shared_event, allow_timing));
  }
}

void MetalEvent::PublishRecordedValue(uint64_t value,
                                      id<MTLCommandBuffer> cmd_buf) {
  if (!allow_timing_) {
    PublishRecordedValue(value);
    return;
  }
  // Hold the lock across the CAS so timing_record_ stays consistent with
  // recorded_value_ for any GetTimingRecord caller. The single-arg
  // PublishRecordedValue does the monotonic-max CAS and tells us whether
  // this value won.
  absl::MutexLock lock(&timing_mu_);
  if (!PublishRecordedValue(value)) {
    return;
  }
  timing_record_.value = value;
  timing_record_.command_buffer = cmd_buf; // strong retain via ARC
}

absl::StatusOr<MetalEvent::RecordedCommandBuffer>
MetalEvent::GetTimingRecord() const {
  if (!allow_timing_) {
    return absl::FailedPreconditionError(
        "MetalEvent::GetTimingRecord: event was not created with timing "
        "enabled.");
  }
  absl::MutexLock lock(&timing_mu_);
  if (timing_record_.command_buffer == nil || timing_record_.value == 0) {
    return absl::FailedPreconditionError(
        "MetalEvent::GetTimingRecord: event has no recorded command buffer; "
        "RecordEvent was not called.");
  }
  return timing_record_;
}

namespace {

absl::Status CommandBufferStatusToStatus(id<MTLCommandBuffer> cmd_buf,
                                         absl::string_view stage) {
  if (cmd_buf.status != MTLCommandBufferStatusError) {
    return absl::OkStatus();
  }
  NSError *error = cmd_buf.error;
  NSString *desc = error == nil ? nil : [error localizedDescription];
  const char *utf8 = desc == nil ? nullptr : [desc UTF8String];
  return absl::InternalError(
      absl::StrCat(stage, ": command buffer entered error state: ",
                   utf8 == nullptr ? "(no error info)" : utf8));
}

} // namespace

absl::StatusOr<absl::Duration>
MetalEvent::ElapsedDurationSince(const MetalEvent &start) const {
  @autoreleasepool {
    TF_ASSIGN_OR_RETURN(RecordedCommandBuffer start_record,
                        start.GetTimingRecord());
    TF_ASSIGN_OR_RETURN(RecordedCommandBuffer stop_record, GetTimingRecord());

    [start_record.command_buffer waitUntilCompleted];
    [stop_record.command_buffer waitUntilCompleted];
    TF_RETURN_IF_ERROR(CommandBufferStatusToStatus(
        start_record.command_buffer, "MetalEvent::Elapsed start"));
    TF_RETURN_IF_ERROR(CommandBufferStatusToStatus(stop_record.command_buffer,
                                                   "MetalEvent::Elapsed stop"));

    // Each event-record cmd_buf has only a single encodeSignalEvent; GPUEndTime
    // of the start cmd_buf marks when the start signal fired, GPUStartTime of
    // the stop cmd_buf marks when the stop signal is about to fire. The
    // interval between those two is the elapsed GPU time.
    CFTimeInterval start_time = start_record.command_buffer.GPUEndTime;
    CFTimeInterval stop_time = stop_record.command_buffer.GPUStartTime;
    if (start_time == 0 || stop_time == 0 || stop_time < start_time) {
      return absl::InternalError(absl::StrCat(
          "MetalEvent::ElapsedDurationSince: invalid GPU timestamps "
          "(start_GPUEndTime=",
          start_time, ", stop_GPUStartTime=", stop_time, ")."));
    }
    return absl::Seconds(stop_time - start_time);
  }
}

void MetalEvent::MarkSignalErrorForValue(
    const std::shared_ptr<MetalEventErrorState> &state, uint64_t value,
    absl::Status status) {
  if (state == nullptr || status.ok())
    return;
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
  if (error_state_->error_value == value && !error_state_->error_status.ok()) {
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
      BOOL signaled =
          [shared_event_ waitUntilSignaledValue:value timeoutMS:kWaitChunkMs];
      if (signaled) {
        return absl::OkStatus();
      }
    }
  }
}

} // namespace metal
} // namespace stream_executor
