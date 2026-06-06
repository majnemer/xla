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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_TIMER_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_TIMER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_stream.h"

namespace stream_executor {
namespace metal {

// EventBasedTimer implementation for Metal. Records a timing-capable
// MetalEvent at Create() and a second one at GetElapsedDuration(); reads
// GPU time off the two underlying MTLCommandBuffers via
// MetalEvent::ElapsedDurationSince. Unlike CudaTimer there is no delay
// kernel — Apple Silicon GPUs don't have a documented analog and the GPU
// timestamps come from each signal cmd_buf's completion metadata directly.
class MetalTimer : public EventBasedTimer {
 public:
  ~MetalTimer() override = default;
  MetalTimer(MetalTimer&&) = default;
  MetalTimer& operator=(MetalTimer&&) = default;

  static absl::StatusOr<MetalTimer> Create(MetalExecutor* executor,
                                           MetalStream* stream);

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

 private:
  MetalTimer(MetalExecutor* executor, MetalStream* stream,
             std::unique_ptr<MetalEvent> start_event,
             std::unique_ptr<MetalEvent> stop_event);

  MetalExecutor* executor_;
  MetalStream* stream_;
  std::unique_ptr<MetalEvent> start_event_;
  std::unique_ptr<MetalEvent> stop_event_;
  bool stopped_ = false;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_TIMER_H_
