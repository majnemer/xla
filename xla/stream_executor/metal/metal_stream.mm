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

#include "xla/stream_executor/metal/metal_stream.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace metal {

struct MetalStreamAsyncErrorState {
  absl::Mutex mu;
  std::optional<absl::Status> first_error;
};

namespace {

absl::Status CommandBufferErrorToStatus(id<MTLCommandBuffer> cmd_buf,
                                        absl::string_view op_name) {
  NSError *error = cmd_buf.error;
  NSString *desc = error == nil ? nil : [error localizedDescription];
  const char *utf8 = desc == nil ? nullptr : [desc UTF8String];
  return absl::InternalError(absl::StrCat(
      op_name, ": ",
      utf8 == nullptr ? "MTLCommandBuffer entered error state." : utf8));
}

void StoreFirstAsyncError(
    const std::shared_ptr<MetalStreamAsyncErrorState> &state,
    absl::Status status) {
  if (status.ok())
    return;
  absl::MutexLock lock(&state->mu);
  if (!state->first_error.has_value()) {
    state->first_error = std::move(status);
  }
}

// Reads the first async error without clearing it. Mirrors CUDA's stream
// poisoning: once a stream has hit an async failure, every subsequent
// peek (and every new submission, gated below) keeps seeing the same
// error until the stream is destroyed.
std::optional<absl::Status> PeekFirstAsyncError(
    const std::shared_ptr<MetalStreamAsyncErrorState> &state) {
  absl::MutexLock lock(&state->mu);
  return state->first_error;
}

// Submit-time gate: returns the recorded first async error if the stream
// is poisoned, otherwise OK. Used at the top of every entry point that
// would otherwise commit a new command buffer onto a stream the CUDA
// driver would have rejected outright.
absl::Status PoisonStatusOrOk(
    const std::shared_ptr<MetalStreamAsyncErrorState> &state) {
  if (auto err = PeekFirstAsyncError(state); err.has_value()) {
    return *std::move(err);
  }
  return absl::OkStatus();
}

} // namespace

MetalStream::MetalStream(
    MetalExecutor *executor, id<MTLCommandQueue> command_queue,
    std::optional<std::variant<StreamPriority, int>> priority,
    std::unique_ptr<MetalEvent> completed_event)
    : StreamCommon(executor, priority),
      async_error_state_(std::make_shared<MetalStreamAsyncErrorState>()),
      executor_(executor), command_queue_(command_queue),
      completed_event_(std::move(completed_event)) {}

MetalStream::~MetalStream() {
  // Mirrors CudaStream::~CudaStream — let the executor remove us from its
  // live-streams set so SynchronizeAllActivity / Memcpy paths don't try to
  // BlockHostUntilDone() on a destroyed stream.
  if (executor_ != nullptr) {
    executor_->DeallocateStream(this);
  }
}

void MetalStream::TrackCommandBufferErrors(id<MTLCommandBuffer> cmd_buf,
                                           absl::string_view op_name) {
  std::shared_ptr<MetalStreamAsyncErrorState> state = async_error_state_;
  std::string op_name_copy(op_name);
  [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    if (completed.status == MTLCommandBufferStatusError) {
      StoreFirstAsyncError(state,
                           CommandBufferErrorToStatus(completed, op_name_copy));
    }
  }];
}

absl::StatusOr<std::unique_ptr<MetalStream>>
MetalStream::Create(MetalExecutor *executor,
                    std::optional<std::variant<StreamPriority, int>> priority) {
  id<MTLDevice> device = executor->device();
  if (device == nil) {
    return absl::FailedPreconditionError(
        "MetalStream::Create: executor has no MTLDevice; was Init() called?");
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    return absl::ResourceExhaustedError(
        "MetalStream::Create: [MTLDevice newCommandQueue] returned nil.");
  }
  TF_ASSIGN_OR_RETURN(auto completed_event, MetalEvent::Create(executor));
  return std::unique_ptr<MetalStream>(
      new MetalStream(executor, queue, priority, std::move(completed_event)));
}

absl::StatusOr<uint64_t>
MetalStream::RecordEventAndReturnValue(MetalEvent *metal_event) {
  if (metal_event == nullptr) {
    return absl::InvalidArgumentError(
        "MetalStream::RecordEventAndReturnValue: null event.");
  }
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  // Reserve under submit_mu_ so value allocation matches commit order;
  // otherwise a waiter could be satisfied by a later signal that's already
  // passed while its own signal cmd_buf has not yet executed.
  uint64_t value;
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> cmd_buf = [command_queue_ commandBuffer];
    if (cmd_buf == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::RecordEvent: commandBuffer returned nil.");
    }
    value = metal_event->ReserveNextRecordedValue();
    [cmd_buf encodeSignalEvent:metal_event->shared_event() value:value];
    // Custom completion handler so we can route the error to the event (so
    // waiters/pollers can detect that the signal never arrived) AND to the
    // stream's first-error state (so BlockHostUntilDone surfaces it).
    std::shared_ptr<MetalEventErrorState> event_error_state =
        metal_event->error_state();
    std::shared_ptr<MetalStreamAsyncErrorState> stream_error_state =
        async_error_state_;
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status == MTLCommandBufferStatusError) {
        absl::Status status =
            CommandBufferErrorToStatus(completed, "MetalStream::RecordEvent");
        MetalEvent::MarkSignalErrorForValue(event_error_state, value, status);
        StoreFirstAsyncError(stream_error_state, std::move(status));
      }
    }];
    [cmd_buf commit];
    tail_buffer_ = cmd_buf;
    // Publish only after the signal cmd_buf is submitted — waiters cannot see
    // a value they're not guaranteed an in-flight signal for. Timing events
    // capture the cmd_buf here so MetalTimer can read GPUStart/EndTime after
    // completion.
    metal_event->PublishRecordedValue(value, cmd_buf);
  }
  return value;
}

absl::StatusOr<uint64_t> MetalStream::RecordCompletedEventValue() {
  return RecordEventAndReturnValue(completed_event_.get());
}

absl::Status MetalStream::RecordCompletedEvent() {
  TF_ASSIGN_OR_RETURN(uint64_t ignored, RecordCompletedEventValue());
  (void)ignored;
  return absl::OkStatus();
}

absl::Status MetalStream::RecordEvent(Event *event) {
  TF_ASSIGN_OR_RETURN(uint64_t ignored, RecordEventAndReturnValue(
                                            static_cast<MetalEvent *>(event)));
  (void)ignored;
  return absl::OkStatus();
}

absl::Status MetalStream::WaitForEventValue(MetalEvent *metal_event,
                                            uint64_t value) {
  if (metal_event == nullptr) {
    return absl::InvalidArgumentError(
        "MetalStream::WaitForEventValue: null event.");
  }
  if (value == 0)
    return absl::OkStatus();
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> cmd_buf = [command_queue_ commandBuffer];
    if (cmd_buf == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::WaitFor(Event*): commandBuffer returned nil.");
    }
    [cmd_buf encodeWaitForEvent:metal_event->shared_event() value:value];
    TrackCommandBufferErrors(cmd_buf, "MetalStream::WaitFor(Event*)");
    [cmd_buf commit];
    tail_buffer_ = cmd_buf;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::WaitFor(Event *event) {
  MetalEvent *metal_event = static_cast<MetalEvent *>(event);
  return WaitForEventValue(metal_event, metal_event->recorded_value());
}

absl::Status MetalStream::WaitFor(Stream *other) {
  if (other == this)
    return absl::OkStatus();
  MetalStream *other_stream = static_cast<MetalStream *>(other);
  // Capture the exact value we recorded. If we re-read recorded_value() after
  // the fact, another concurrent WaitFor on the same peer could advance
  // completed_event_'s recorded value and make this stream wait on a newer
  // (later) signal than the one we actually triggered.
  TF_ASSIGN_OR_RETURN(uint64_t value,
                      other_stream->RecordCompletedEventValue());
  return WaitForEventValue(other_stream->completed_event_.get(), value);
}

absl::Status MetalStream::Memcpy(DeviceAddressBase *gpu_dst,
                                 const void *host_src, uint64_t size) {
  if (size == 0)
    return absl::OkStatus();

  if (gpu_dst == nullptr || gpu_dst->opaque() == nullptr ||
      host_src == nullptr) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (H2D): null pointer.");
  }

  auto resolved = executor_->allocator()->Resolve(gpu_dst->opaque());
  if (!resolved.has_value()) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (H2D): destination not owned by this allocator.");
  }

  id<MTLBuffer> dst_buffer = resolved->buffer;
  const NSUInteger dst_offset = static_cast<NSUInteger>(resolved->offset);

  if (dst_buffer.storageMode != MTLStorageModeShared) {
    return absl::FailedPreconditionError(
        "MetalStream::Memcpy(H2D): async host-copy path requires Shared "
        "storage on Apple Silicon.");
  }

  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));

  id<MTLSharedEvent> copy_done = [executor_->device() newSharedEvent];
  if (copy_done == nil) {
    return absl::ResourceExhaustedError(
        "MetalStream::Memcpy(H2D): newSharedEvent returned nil.");
  }

  constexpr uint64_t kCopyDoneValue = 1;
  std::shared_ptr<MetalStreamAsyncErrorState> state = async_error_state_;

  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> copy_cmd = [command_queue_ commandBuffer];
    if (copy_cmd == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy(H2D): copy commandBuffer returned nil.");
    }

    id<MTLCommandBuffer> wait_cmd = [command_queue_ commandBuffer];
    if (wait_cmd == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy(H2D): wait commandBuffer returned nil.");
    }

    TrackCommandBufferErrors(copy_cmd, "MetalStream::Memcpy(H2D copy)");
    TrackCommandBufferErrors(wait_cmd, "MetalStream::Memcpy(H2D wait)");

    [copy_cmd addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status == MTLCommandBufferStatusCompleted) {
        void *dst = static_cast<char *>([dst_buffer contents]) + dst_offset;
        std::memcpy(dst, host_src, size);
      }

      // Always signal so the stream cannot deadlock. Errors are surfaced
      // through TrackCommandBufferErrors / BlockHostUntilDone. The signal
      // publishes the memcpy to the GPU's event wait; Shared storage keeps it
      // coherent.
      copy_done.signaledValue = kCopyDoneValue;
    }];

    [wait_cmd encodeWaitForEvent:copy_done value:kCopyDoneValue];

    [copy_cmd commit];
    [wait_cmd commit];

    tail_buffer_ = wait_cmd;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(void *host_dst,
                                 const DeviceAddressBase &gpu_src,
                                 uint64_t size) {
  if (size == 0)
    return absl::OkStatus();
  if (host_dst == nullptr || gpu_src.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (D2H): null pointer.");
  }
  auto resolved = executor_->allocator()->Resolve(gpu_src.opaque());
  if (!resolved.has_value()) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (D2H): source not owned by this allocator.");
  }
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  id<MTLBuffer> staging =
      [executor_->device() newBufferWithLength:size
                                       options:MTLResourceStorageModeShared];
  if (staging == nil) {
    return absl::ResourceExhaustedError(
        "MetalStream::Memcpy (D2H): staging allocation failed.");
  }
  // Two-cmd_buf pattern: completion-handler memcpy + GPU-side wait, so
  // subsequent cmd_bufs see the host write (not just the blit completion).
  id<MTLSharedEvent> copy_done = [executor_->device() newSharedEvent];
  if (copy_done == nil) {
    return absl::ResourceExhaustedError(
        "MetalStream::Memcpy (D2H): newSharedEvent returned nil.");
  }
  constexpr uint64_t kCopyDoneValue = 1;
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> copy_cmd = [command_queue_ commandBuffer];
    if (copy_cmd == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy (D2H): copy commandBuffer returned nil.");
    }
    id<MTLCommandBuffer> wait_cmd = [command_queue_ commandBuffer];
    if (wait_cmd == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy (D2H): wait commandBuffer returned nil.");
    }
    id<MTLBlitCommandEncoder> blit = [copy_cmd blitCommandEncoder];
    if (blit == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy (D2H): blitCommandEncoder returned nil.");
    }
    [blit copyFromBuffer:resolved->buffer
             sourceOffset:resolved->offset
                 toBuffer:staging
        destinationOffset:0
                     size:size];
    [blit endEncoding];
    TrackCommandBufferErrors(copy_cmd, "MetalStream::Memcpy(D2H)(copy)");
    TrackCommandBufferErrors(wait_cmd, "MetalStream::Memcpy(D2H)(wait)");
    [copy_cmd addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status == MTLCommandBufferStatusCompleted) {
        std::memcpy(host_dst, [staging contents], size);
      }
      // Always signal so the wait cmd_buf cannot deadlock if the blit
      // errored before completing.
      copy_done.signaledValue = kCopyDoneValue;
    }];
    [wait_cmd encodeWaitForEvent:copy_done value:kCopyDoneValue];
    [copy_cmd commit];
    [wait_cmd commit];
    // Tail = wait_cmd so BlockHostUntilDone joins on the host memcpy too,
    // and so a later submission under submit_mu_ is FIFO-ordered after the
    // host write has happened.
    tail_buffer_ = wait_cmd;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(DeviceAddressBase *gpu_dst,
                                 const DeviceAddressBase &gpu_src,
                                 uint64_t size) {
  if (size == 0)
    return absl::OkStatus();
  if (gpu_dst == nullptr || gpu_dst->opaque() == nullptr ||
      gpu_src.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (D2D): null pointer.");
  }
  MetalAllocator *allocator = executor_->allocator();
  auto src_resolved = allocator->Resolve(gpu_src.opaque());
  auto dst_resolved = allocator->Resolve(gpu_dst->opaque());
  if (!src_resolved.has_value() || !dst_resolved.has_value()) {
    return absl::InvalidArgumentError(
        "MetalStream::Memcpy (D2D): endpoint not owned by this allocator.");
  }
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> cmd_buf = [command_queue_ commandBuffer];
    if (cmd_buf == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy (D2D): commandBuffer returned nil.");
    }
    id<MTLBlitCommandEncoder> blit = [cmd_buf blitCommandEncoder];
    if (blit == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::Memcpy (D2D): blitCommandEncoder returned nil.");
    }
    [blit copyFromBuffer:src_resolved->buffer
             sourceOffset:src_resolved->offset
                 toBuffer:dst_resolved->buffer
        destinationOffset:dst_resolved->offset
                     size:size];
    [blit endEncoding];
    TrackCommandBufferErrors(cmd_buf, "MetalStream::Memcpy(D2D)");
    [cmd_buf commit];
    tail_buffer_ = cmd_buf;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  // Two-cmd_buf design: callback_cmd's completion handler runs the host
  // callback and signals callback_done; wait_cmd encodes a GPU wait on
  // it. submit_mu_ keeps later cmd_bufs FIFO-ordered after wait_cmd.
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  auto *heap_cb =
      new absl::AnyInvocable<absl::Status() &&>(std::move(callback));
  id<MTLSharedEvent> callback_done = [executor_->device() newSharedEvent];
  if (callback_done == nil) {
    delete heap_cb;
    return absl::ResourceExhaustedError(
        "MetalStream::DoHostCallbackWithStatus: newSharedEvent returned nil.");
  }
  constexpr uint64_t kCallbackDoneValue = 1;
  std::shared_ptr<MetalStreamAsyncErrorState> state = async_error_state_;

  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> callback_cmd = [command_queue_ commandBuffer];
    if (callback_cmd == nil) {
      delete heap_cb;
      return absl::ResourceExhaustedError(
          "MetalStream::DoHostCallbackWithStatus: callback commandBuffer "
          "returned nil.");
    }
    id<MTLCommandBuffer> wait_cmd = [command_queue_ commandBuffer];
    if (wait_cmd == nil) {
      delete heap_cb;
      return absl::ResourceExhaustedError(
          "MetalStream::DoHostCallbackWithStatus: wait commandBuffer returned "
          "nil.");
    }
    TrackCommandBufferErrors(callback_cmd,
                             "MetalStream::DoHostCallbackWithStatus(callback)");
    TrackCommandBufferErrors(wait_cmd,
                             "MetalStream::DoHostCallbackWithStatus(wait)");
    [callback_cmd addCompletedHandler:^(id<MTLCommandBuffer> /*completed*/) {
      // Always signal callback_done, even if the callback returns an error,
      // so subsequent waiters cannot deadlock.
      absl::Status callback_status = std::move(*heap_cb)();
      delete heap_cb;
      StoreFirstAsyncError(state, std::move(callback_status));
      callback_done.signaledValue = kCallbackDoneValue;
    }];
    [wait_cmd encodeWaitForEvent:callback_done value:kCallbackDoneValue];
    [callback_cmd commit];
    [wait_cmd commit];
    // Tail = wait_cmd so BlockHostUntilDone joins on the CPU callback too.
    tail_buffer_ = wait_cmd;
  }
  return absl::OkStatus();
}

absl::Status
MetalStream::LaunchKernel(const ThreadDim &thread_dims,
                          const BlockDim &block_dims,
                          const std::optional<ClusterDim> &cluster_dims,
                          void *function, absl::string_view name, void **args,
                          int64_t shmem_bytes, bool /*use_pdl*/) {
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError(
        "MetalStream::LaunchKernel: cluster dimensions are not supported on "
        "Metal.");
  }
  auto *kernel = static_cast<MetalKernel *>(function);
  if (kernel == nullptr || kernel->pipeline_state() == nil) {
    return absl::InvalidArgumentError(
        "MetalStream::LaunchKernel: null kernel or pipeline state.");
  }

  const int64_t total_threads = thread_dims.x * thread_dims.y * thread_dims.z;
  const int64_t pso_limit =
      static_cast<int64_t>([kernel->pipeline_state() maxTotalThreadsPerThreadgroup]);
  if (total_threads > pso_limit) {
    return absl::FailedPreconditionError(absl::StrCat(
        "MetalStream::LaunchKernel: kernel ", name, " block size ",
        total_threads, " exceeds pipeline state maxTotalThreadsPerThreadgroup ",
        pso_limit, "."));
  }

  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> cmd_buf = [command_queue_ commandBuffer];
    if (cmd_buf == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::LaunchKernel: commandBuffer returned nil.");
    }
    if (!name.empty()) {
      cmd_buf.label = [[NSString alloc] initWithBytes:name.data()
                                               length:name.size()
                                             encoding:NSUTF8StringEncoding];
    }
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    if (encoder == nil) {
      return absl::ResourceExhaustedError(
          "MetalStream::LaunchKernel: computeCommandEncoder returned nil.");
    }
    [encoder setComputePipelineState:kernel->pipeline_state()];
    MetalAllocator *allocator = executor_->allocator();
    const unsigned arity = kernel->Arity();
    for (unsigned i = 0; i < arity; ++i) {
      void *arg_ptr = *static_cast<void *const *>(args[i]);
      if (arg_ptr == nullptr) {
        [encoder endEncoding];
        return absl::InvalidArgumentError(absl::StrCat(
            "MetalStream::LaunchKernel: argument ", i, " is null."));
      }
      auto resolved = allocator->Resolve(arg_ptr);
      if (!resolved.has_value()) {
        [encoder endEncoding];
        return absl::InvalidArgumentError(
            absl::StrCat("MetalStream::LaunchKernel: argument ", i,
                         " is not owned by this executor's allocator."));
      }
      [encoder setBuffer:resolved->buffer offset:resolved->offset atIndex:i];
    }
    if (shmem_bytes > 0) {
      [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(shmem_bytes)
                                  atIndex:arity];
    }
    MTLSize threadgroups =
        MTLSizeMake(block_dims.x, block_dims.y, block_dims.z);
    MTLSize threadsPerThreadgroup =
        MTLSizeMake(thread_dims.x, thread_dims.y, thread_dims.z);
    [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    TrackCommandBufferErrors(cmd_buf, "MetalStream::LaunchKernel");
    [cmd_buf commit];
    tail_buffer_ = cmd_buf;
  }
  return absl::OkStatus();
}

absl::StatusOr<id<MTLCommandBuffer>> MetalStream::EncodeWithCommandBuffer(
    absl::string_view label, absl::string_view op_name,
    absl::FunctionRef<absl::StatusOr<id<MTLCommandBuffer>>(
        id<MTLCommandBuffer> cmd_buf)>
        encode) {
  TF_RETURN_IF_ERROR(PoisonStatusOrOk(async_error_state_));
  @autoreleasepool {
    absl::MutexLock lock(&submit_mu_);
    id<MTLCommandBuffer> cmd_buf = [command_queue_ commandBuffer];
    if (cmd_buf == nil) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "MetalStream::EncodeWithCommandBuffer: commandBuffer returned nil "
          "for '", op_name, "'."));
    }
    if (!label.empty()) {
      cmd_buf.label = [[NSString alloc] initWithBytes:label.data()
                                               length:label.size()
                                             encoding:NSUTF8StringEncoding];
    }
    TF_ASSIGN_OR_RETURN(id<MTLCommandBuffer> tail, encode(cmd_buf));
    if (tail == nil) {
      return absl::InternalError(absl::StrCat(
          "MetalStream::EncodeWithCommandBuffer: encode callback returned nil "
          "tail for '", op_name, "'."));
    }
    TrackCommandBufferErrors(tail, op_name);
    [tail commit];
    tail_buffer_ = tail;
    return tail;
  }
}

absl::Status MetalStream::BlockHostUntilDone() {
  id<MTLCommandBuffer> tail = nil;
  {
    absl::MutexLock lock(&submit_mu_);
    tail = tail_buffer_;
  }
  if (tail != nil) {
    [tail waitUntilCompleted];
  }
  // Return the first async error seen anywhere on this stream, not just on
  // the current tail. Sticky: future calls keep returning the same error.
  if (auto async_error = PeekFirstAsyncError(async_error_state_);
      async_error.has_value()) {
    return *std::move(async_error);
  }
  // Fallback for the case where a completion handler did not get to record
  // before waitUntilCompleted returned.
  if (tail != nil && tail.status == MTLCommandBufferStatusError) {
    return CommandBufferErrorToStatus(tail, "MetalStream::BlockHostUntilDone");
  }
  return absl::OkStatus();
}

} // namespace metal
} // namespace stream_executor
