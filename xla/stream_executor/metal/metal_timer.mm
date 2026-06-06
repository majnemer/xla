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

#include "xla/stream_executor/metal/metal_timer.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace metal {

MetalTimer::MetalTimer(MetalExecutor* executor, MetalStream* stream,
                       std::unique_ptr<MetalEvent> start_event,
                       std::unique_ptr<MetalEvent> stop_event)
    : executor_(executor),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

absl::StatusOr<MetalTimer> MetalTimer::Create(MetalExecutor* executor,
                                              MetalStream* stream) {
  if (executor == nullptr || stream == nullptr) {
    return absl::InvalidArgumentError(
        "MetalTimer::Create: executor and stream must be non-null.");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MetalEvent> start_event,
                      MetalEvent::Create(executor, /*allow_timing=*/true));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MetalEvent> stop_event,
                      MetalEvent::Create(executor, /*allow_timing=*/true));
  TF_RETURN_IF_ERROR(stream->RecordEvent(start_event.get()));
  return MetalTimer(executor, stream, std::move(start_event),
                    std::move(stop_event));
}

absl::StatusOr<absl::Duration> MetalTimer::GetElapsedDuration() {
  if (stopped_) {
    return absl::FailedPreconditionError(
        "MetalTimer::GetElapsedDuration called on a stopped timer.");
  }
  TF_RETURN_IF_ERROR(stream_->RecordEvent(stop_event_.get()));
  stopped_ = true;
  return stop_event_->ElapsedDurationSince(*start_event_);
}

}  // namespace metal
}  // namespace stream_executor
