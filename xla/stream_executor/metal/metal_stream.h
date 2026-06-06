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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace metal {

struct MetalStreamAsyncErrorState;

// Per-stream wrapper around an MTLCommandQueue.
//
// Stream order is defined by `submit_mu_`: every cmd_buf creation, encoding,
// commit, and tail update happens under this lock, in submission order. The
// last committed cmd_buf is the "tail"; BlockHostUntilDone waits on it.
//
// Errors raised inside command-buffer completion handlers (or the callback
// body of DoHostCallback) are funneled through `async_error_state_` and
// surfaced by BlockHostUntilDone — not lost.
class MetalStream : public StreamCommon {
 public:
  static absl::StatusOr<std::unique_ptr<MetalStream>> Create(
      MetalExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);
  ~MetalStream() override;

  // Records this stream's completion event at the current tail. Used by
  // WaitFor(Stream*) on a peer stream to synchronize with us.
  absl::Status RecordCompletedEvent();

  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;

  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src, uint64_t size) override;

  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

  absl::Status BlockHostUntilDone() override;

  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name,
                            void** args, int64_t shmem_bytes,
                            bool use_pdl) override;

  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override {
    return executor_->CreateEventBasedTimer(this, use_delay_kernel);
  }

  // Underlying MTLCommandQueue. Available only inside .mm consumers.
  id<MTLCommandQueue> command_queue() const { return command_queue_; }

  // Runs `encode` on a fresh MTLCommandBuffer created from this stream's
  // queue, under the stream's submission lock so ordering with concurrent
  // LaunchKernel calls is preserved.
  //
  // The callback's job is to encode work (and, for wrappers like
  // MPSCommandBuffer, run any internal commitAndContinue) and return the
  // command buffer the stream should track as the new tail — which may
  // differ from the input cmd_buf if MPSGraph or similar split the encoded
  // work across buffers. The callback must NOT commit the returned buffer:
  // the stream installs the async-error completion handler and then commits
  // it once. `label` is applied to the initial cmd_buf for diagnostics;
  // `op_name` flows into the async-error tracker.
  absl::StatusOr<id<MTLCommandBuffer>> EncodeWithCommandBuffer(
      absl::string_view label, absl::string_view op_name,
      absl::FunctionRef<absl::StatusOr<id<MTLCommandBuffer>>(
          id<MTLCommandBuffer> cmd_buf)>
          encode);

 private:
  MetalStream(MetalExecutor* executor, id<MTLCommandQueue> command_queue,
              std::optional<std::variant<StreamPriority, int>> priority,
              std::unique_ptr<MetalEvent> completed_event);

  // Internal helpers. RecordEventAndReturnValue submits the signal and
  // returns the value to use, so WaitFor(Stream*) can lock in the exact value
  // it caused to be recorded (instead of re-reading recorded_value() later
  // and racing with another WaitFor on the same peer).
  absl::StatusOr<uint64_t> RecordEventAndReturnValue(MetalEvent* event);
  absl::StatusOr<uint64_t> RecordCompletedEventValue();
  absl::Status WaitForEventValue(MetalEvent* event, uint64_t value);

  // Adds a completion handler that records the cmd_buf's error (if any) into
  // `async_error_state_`. Always installed before commit. The handler captures
  // a shared_ptr to async_error_state_ so it cannot UAF the MetalStream.
  void TrackCommandBufferErrors(id<MTLCommandBuffer> cmd_buf,
                                absl::string_view op_name);

  // Serializes cmd_buf creation, encoding, commit, and tail update on this
  // stream. Defines the host-side submission order.
  absl::Mutex submit_mu_;
  id<MTLCommandBuffer> tail_buffer_ ABSL_GUARDED_BY(submit_mu_) = nil;

  // Shared so cmd_buf completion handlers cannot use-after-free the stream.
  std::shared_ptr<MetalStreamAsyncErrorState> async_error_state_;

  MetalExecutor* executor_;
  id<MTLCommandQueue> command_queue_;

  // Per-stream completion event used to implement WaitFor(Stream*).
  // RecordCompletedEvent() entrains a signal on this event at the current
  // tail; a peer stream WaitFor's the same event.
  std::unique_ptr<MetalEvent> completed_event_;

  MetalStream(const MetalStream&) = delete;
  MetalStream& operator=(const MetalStream&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
