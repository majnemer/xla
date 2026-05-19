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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::metal {
namespace {

// A trivial fill-by-thread-id kernel. Each thread writes its global thread
// index into the output buffer, so a verified result proves: MSL compile,
// PSO build, buffer binding, dispatch, and GPU<->host data path all worked.
constexpr absl::string_view kFillKernelSource = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void fill_kernel(device int* out [[buffer(0)]],
                        uint tid [[thread_position_in_grid]]) {
  out[tid] = int(tid);
}
)msl";

TEST(MetalSmokeTest, FillKernelEndToEnd) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  ASSERT_GE(platform->VisibleDeviceCount(), 1)
      << "No Metal devices visible — this test requires Apple Silicon.";

  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());

  constexpr int kThreadsPerGroup = 32;
  constexpr int kThreadgroups = 8;
  constexpr int kTotal = kThreadsPerGroup * kThreadgroups;
  constexpr int64_t kBytes = kTotal * sizeof(int);

  KernelLoaderSpec spec = KernelLoaderSpec::CreateMslSourceInMemorySpec(
      kFillKernelSource, "fill_kernel", /*arity=*/1);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(spec));

  DeviceAddressBase out_buf = executor->Allocate(kBytes, /*memory_space=*/0);
  ASSERT_NE(out_buf.opaque(), nullptr);

  // Pre-fill with sentinel so we can detect untouched slots if the kernel
  // doesn't actually run.
  std::vector<int> sentinel(kTotal, -1);
  TF_ASSERT_OK(stream->Memcpy(&out_buf, sentinel.data(), kBytes));

  KernelArgsPackedArray args(/*num_args=*/1);
  args.add_argument(out_buf);

  TF_ASSERT_OK(kernel->Launch(ThreadDim(kThreadsPerGroup),
                              BlockDim(kThreadgroups),
                              /*cluster_dims=*/std::nullopt, stream.get(),
                              args));

  std::vector<int> host(kTotal, -2);
  TF_ASSERT_OK(stream->Memcpy(host.data(), out_buf, kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ(host[i], i) << "i=" << i;
  }

  executor->Deallocate(&out_buf);
}

// Returns a Metal executor for device 0, asserting Metal is available.
StreamExecutor* GetMetalExecutorOrSkip() {
  auto platform = PlatformManager::PlatformWithName("METAL");
  if (!platform.ok()) {
    ADD_FAILURE() << "METAL platform not registered: " << platform.status();
    return nullptr;
  }
  if ((*platform)->VisibleDeviceCount() < 1) {
    ADD_FAILURE() << "No Metal devices visible.";
    return nullptr;
  }
  auto executor = (*platform)->ExecutorForDevice(0);
  if (!executor.ok()) {
    ADD_FAILURE() << "ExecutorForDevice(0) failed: " << executor.status();
    return nullptr;
  }
  return *executor;
}

// Cross-stream synchronization: stream A writes a pattern via a kernel and
// records an event; stream B waits on that event and copies the buffer
// device-to-device. Without the event wait, B's blit can race ahead of A's
// fill and observe stale data.
TEST(MetalSmokeTest, EventSyncBetweenStreams) {
  StreamExecutor* executor = GetMetalExecutorOrSkip();
  ASSERT_NE(executor, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream_a,
                          executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream_b,
                          executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Event> event,
                          executor->CreateEvent());

  constexpr int kTotal = 256;
  constexpr int64_t kBytes = kTotal * sizeof(int);

  KernelLoaderSpec spec = KernelLoaderSpec::CreateMslSourceInMemorySpec(
      kFillKernelSource, "fill_kernel", /*arity=*/1);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(spec));

  DeviceAddressBase src = executor->Allocate(kBytes, /*memory_space=*/0);
  DeviceAddressBase dst = executor->Allocate(kBytes, /*memory_space=*/0);
  ASSERT_NE(src.opaque(), nullptr);
  ASSERT_NE(dst.opaque(), nullptr);

  // A: fill src by tid, record event.
  KernelArgsPackedArray fill_args(/*num_args=*/1);
  fill_args.add_argument(src);
  TF_ASSERT_OK(kernel->Launch(ThreadDim(32), BlockDim(kTotal / 32),
                              /*cluster_dims=*/std::nullopt, stream_a.get(),
                              fill_args));
  TF_ASSERT_OK(stream_a->RecordEvent(event.get()));

  // B: wait on event, D2D copy src -> dst.
  TF_ASSERT_OK(stream_b->WaitFor(event.get()));
  TF_ASSERT_OK(stream_b->Memcpy(&dst, src, kBytes));

  std::vector<int> host(kTotal, -1);
  TF_ASSERT_OK(stream_b->Memcpy(host.data(), dst, kBytes));
  TF_ASSERT_OK(stream_b->BlockHostUntilDone());

  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ(host[i], i) << "i=" << i;
  }
  EXPECT_EQ(event->PollForStatus(), Event::Status::kComplete);
  TF_ASSERT_OK(event->Synchronize());

  executor->Deallocate(&src);
  executor->Deallocate(&dst);
}

// D2D blit between two device buffers, with neither side requiring a kernel
// launch. Exercises MetalStream::Memcpy(DeviceAddressBase*, DeviceAddressBase&)
// and the dual-Resolve path in the allocator.
TEST(MetalSmokeTest, DeviceToDeviceMemcpy) {
  StreamExecutor* executor = GetMetalExecutorOrSkip();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());

  constexpr int kTotal = 1024;
  constexpr int64_t kBytes = kTotal * sizeof(int);
  std::vector<int> pattern(kTotal);
  for (int i = 0; i < kTotal; ++i) pattern[i] = i * 7 + 3;

  DeviceAddressBase src = executor->Allocate(kBytes, /*memory_space=*/0);
  DeviceAddressBase dst = executor->Allocate(kBytes, /*memory_space=*/0);
  ASSERT_NE(src.opaque(), nullptr);
  ASSERT_NE(dst.opaque(), nullptr);

  TF_ASSERT_OK(stream->Memcpy(&src, pattern.data(), kBytes));
  TF_ASSERT_OK(stream->Memcpy(&dst, src, kBytes));

  std::vector<int> host(kTotal, 0);
  TF_ASSERT_OK(stream->Memcpy(host.data(), dst, kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(host, pattern);

  executor->Deallocate(&src);
  executor->Deallocate(&dst);
}

// Host callback fires after the preceding stream work completes. The
// completion handler should observe the kernel's writes via unified memory
// because Apple's MTLCommandBuffer guarantees handlers run before the
// buffer's status flips to Completed, and we then BlockHostUntilDone.
TEST(MetalSmokeTest, HostCallbackAfterKernel) {
  StreamExecutor* executor = GetMetalExecutorOrSkip();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());

  constexpr int kTotal = 64;
  constexpr int64_t kBytes = kTotal * sizeof(int);

  KernelLoaderSpec spec = KernelLoaderSpec::CreateMslSourceInMemorySpec(
      kFillKernelSource, "fill_kernel", /*arity=*/1);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(spec));

  DeviceAddressBase buf = executor->Allocate(kBytes, /*memory_space=*/0);
  ASSERT_NE(buf.opaque(), nullptr);

  KernelArgsPackedArray args(/*num_args=*/1);
  args.add_argument(buf);
  TF_ASSERT_OK(kernel->Launch(ThreadDim(32), BlockDim(kTotal / 32),
                              /*cluster_dims=*/std::nullopt, stream.get(),
                              args));

  // The callback reads the kernel's output via the buffer's host pointer
  // (Shared mode == unified memory on Apple Silicon). We record the read
  // result into a heap-shared sum so this test thread can verify ordering.
  auto sum = std::make_shared<std::atomic<int64_t>>(0);
  auto fired = std::make_shared<std::atomic<bool>>(false);
  TF_ASSERT_OK(stream->DoHostCallback(
      [sum, fired, buf, kTotal]() mutable {
        const int* p = static_cast<const int*>(buf.opaque());
        int64_t s = 0;
        for (int i = 0; i < kTotal; ++i) s += p[i];
        sum->store(s);
        fired->store(true);
      }));

  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_TRUE(fired->load());
  // Sum of 0..63 = 63*64/2 = 2016.
  EXPECT_EQ(sum->load(), 2016);

  executor->Deallocate(&buf);
}

// HostMemoryAllocate returns memory that is host-readable and that
// Stream::Memcpy can use as either source or destination via Resolve(),
// thanks to unified memory.
TEST(MetalSmokeTest, HostMemoryAllocateRoundTrip) {
  StreamExecutor* executor = GetMetalExecutorOrSkip();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());

  constexpr int kTotal = 128;
  constexpr int64_t kBytes = kTotal * sizeof(int);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> host_alloc,
                          executor->HostMemoryAllocate(kBytes));
  ASSERT_NE(host_alloc->address().opaque(), nullptr);

  DeviceAddressBase dev = executor->Allocate(kBytes, /*memory_space=*/0);
  ASSERT_NE(dev.opaque(), nullptr);

  // Write to host memory via the raw pointer, blit into device buffer, read
  // back via a separate stream Memcpy into a plain vector.
  int* host_ptr = static_cast<int*>(host_alloc->address().opaque());
  for (int i = 0; i < kTotal; ++i) host_ptr[i] = i * 5;

  TF_ASSERT_OK(stream->Memcpy(&dev, host_ptr, kBytes));

  std::vector<int> readback(kTotal, 0);
  TF_ASSERT_OK(stream->Memcpy(readback.data(), dev, kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ(readback[i], i * 5) << "i=" << i;
  }

  executor->Deallocate(&dev);
}

}  // namespace
}  // namespace stream_executor::metal
