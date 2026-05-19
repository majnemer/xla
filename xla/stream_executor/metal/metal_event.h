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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_

#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {
namespace metal {

class MetalExecutor;

struct MetalEventErrorState;

// A StreamExecutor Event implemented with MTLSharedEvent.
//
// recorded_value_ is the latest value whose signal command buffer has been
// successfully submitted. Waiters/pollers must never observe a value that has
// only been reserved but not submitted.
class MetalEvent final : public Event {
 public:
  static absl::StatusOr<std::unique_ptr<MetalEvent>> Create(
      MetalExecutor* executor);

  ~MetalEvent() override;

  MetalEvent(const MetalEvent&) = delete;
  MetalEvent& operator=(const MetalEvent&) = delete;

  id<MTLSharedEvent> shared_event() const { return shared_event_; }

  uint64_t recorded_value() const {
    return recorded_value_.load(std::memory_order_acquire);
  }

  uint64_t ReserveNextRecordedValue() {
    return next_value_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

  // Monotonic-max: never moves recorded_value_ backwards. When the same
  // event is recorded on two streams concurrently (each holds its own submit
  // lock), reserve order is well-defined (next_value_ is atomic) but publish
  // order is not — without max, a smaller value published last could
  // overwrite a larger one and let a later waiter pass on a signal that
  // hasn't run yet.
  void PublishRecordedValue(uint64_t value) {
    uint64_t old = recorded_value_.load(std::memory_order_relaxed);
    while (old < value &&
           !recorded_value_.compare_exchange_weak(
               old, value, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
  }

  // Called by command-buffer completion handlers if the command buffer that
  // was supposed to signal `value` fails.
  void MarkSignalErrorForValue(uint64_t value, absl::Status status);

  // Safe to capture in Obj-C command-buffer completion handlers. This prevents
  // a use-after-free if the MetalEvent object is destroyed before the handler
  // runs.
  std::shared_ptr<MetalEventErrorState> error_state() const {
    return error_state_;
  }

  static void MarkSignalErrorForValue(
      const std::shared_ptr<MetalEventErrorState>& state, uint64_t value,
      absl::Status status);

  Event::Status PollForStatus() override;
  absl::Status Synchronize() override;

  uint64_t AllocateNextRecordedValue() = delete;

 private:
  explicit MetalEvent(id<MTLSharedEvent> shared_event);

  std::optional<absl::Status> ErrorForValue(uint64_t value) const;

  __strong id<MTLSharedEvent> shared_event_;

  std::atomic<uint64_t> next_value_{0};
  std::atomic<uint64_t> recorded_value_{0};

  std::shared_ptr<MetalEventErrorState> error_state_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_
