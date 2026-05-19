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

// End-to-end validation that the existing Thunk runtime composes with the
// Metal StreamExecutor surface. If this passes, the architectural call to
// reuse GpuExecutable + Thunk runtime (rather than building MetalExecutable
// from scratch) is confirmed: KernelThunk only touches the abstract
// se::Kernel / se::Stream APIs, both of which we implemented in P1.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/executable_run_options.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace metal {
namespace {

namespace se = ::stream_executor;

// Trivial single-thread add-int32 kernel. Single threadgroup of a single
// thread; tid is always 0 so we just compute c[0] = a[0] + b[0].
constexpr absl::string_view kAddI32Source = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void add_i32(device int* a [[buffer(0)]],
                    device int* b [[buffer(1)]],
                    device int* c [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
  c[tid] = a[tid] + b[tid];
}
)msl";

// Increment-by-one kernel — used to verify WhileThunk re-runs its body the
// expected number of times.
constexpr absl::string_view kIncrementSource = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void increment(device int* counter [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
  counter[tid] += 1;
}
)msl";

// Returns the Metal executor for device 0; ADD_FAILUREs on absence so the
// test reports cleanly rather than crashing on a missing GPU.
se::StreamExecutor* GetMetalExecutorOrFail() {
  auto platform = se::PlatformManager::PlatformWithName("METAL");
  if (!platform.ok()) {
    ADD_FAILURE() << "METAL platform not registered: " << platform.status();
    return nullptr;
  }
  if ((*platform)->VisibleDeviceCount() < 1) {
    ADD_FAILURE() << "No Metal devices visible — Apple Silicon required.";
    return nullptr;
  }
  auto executor = (*platform)->ExecutorForDevice(0);
  if (!executor.ok()) {
    ADD_FAILURE() << "ExecutorForDevice(0) failed: " << executor.status();
    return nullptr;
  }
  return *executor;
}

TEST(MetalKernelThunkTest, AddI32EndToEnd) {
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName("METAL"));
  ASSERT_GE(platform->VisibleDeviceCount(), 1)
      << "No Metal devices visible — this test requires Apple Silicon.";
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  // Allocate three single-element int32 device buffers.
  se::DeviceAddress<int32_t> a_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  se::DeviceAddress<int32_t> b_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  se::DeviceAddress<int32_t> c_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  ASSERT_NE(a_dev.opaque(), nullptr);
  ASSERT_NE(b_dev.opaque(), nullptr);
  ASSERT_NE(c_dev.opaque(), nullptr);

  // Initialize: a=1, b=2, c=0.
  int32_t val_a = 1, val_b = 2, zero = 0;
  TF_ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memcpy(&c_dev, &zero, sizeof(int32_t)));

  // BufferAllocations describes the logical buffer layout that KernelThunk's
  // KernelArgument slices reference. Each allocation maps to one device
  // address in the BufferAllocations array.
  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };

  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  auto thunk = std::make_unique<gpu::KernelThunk>(
      gpu::Thunk::ThunkInfo(),
      /*kernel_name=*/"add_i32",
      emitters::KernelArguments({arg_a, arg_b, arg_c}),
      gpu::LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  // Initialize: KernelThunk reads src.text and invokes
  // stream_executor_util::CreateKernel, which now dispatches on platform id
  // (CreateMslSourceInMemorySpec for Metal vs CreateCudaPtxInMemorySpec for
  // CUDA).
  // ExecutableSource::text is an absl::string_view — assign the literal
  // directly so the view points at static storage. Assigning a temporary
  // std::string would leave the view dangling.
  gpu::Thunk::ExecutableSource src;
  src.text = kAddI32Source;

  se::StreamExecutorAddressAllocator allocator(executor);
  gpu::BufferAllocations buffer_allocations({a_dev, b_dev, c_dev},
                                             /*device_ordinal=*/0, &allocator);

  gpu::Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.src = src;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &buffer_allocations;
  TF_ASSERT_OK(thunk->Initialize(init_params));

  // Execute: KernelThunk packs the per-slice DeviceAddressBases and calls
  // se::Kernel::Launch — which for Metal dispatches through
  // MetalKernel::Launch -> MetalStream::LaunchKernel.
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = gpu::Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory_callbacks=*/nullptr,
      /*async_collective_memory_id=*/{});
  TF_ASSERT_OK(thunk->ExecuteOnStream(execute_params));

  // Read back and verify.
  int32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 3);

  executor->Deallocate(&a_dev);
  executor->Deallocate(&b_dev);
  executor->Deallocate(&c_dev);
}

// DeviceToDeviceCopyThunk uses the abstract `stream->Memcpy(dst, src, n)`
// API, which on Metal goes through MetalStream's blit-encoder D2D path. No
// kernel involved.
TEST(MetalKernelThunkTest, DeviceToDeviceCopyThunk) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  constexpr int kCount = 32;
  constexpr int64_t kBytes = kCount * sizeof(int32_t);

  se::DeviceAddress<int32_t> src_dev =
      executor->AllocateArray<int32_t>(kCount, /*memory_space=*/0);
  se::DeviceAddress<int32_t> dst_dev =
      executor->AllocateArray<int32_t>(kCount, /*memory_space=*/0);
  ASSERT_NE(src_dev.opaque(), nullptr);
  ASSERT_NE(dst_dev.opaque(), nullptr);

  std::vector<int32_t> pattern(kCount);
  for (int i = 0; i < kCount; ++i) pattern[i] = i * 11 - 7;
  TF_ASSERT_OK(stream->Memcpy(&src_dev, pattern.data(), kBytes));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/kBytes, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/kBytes, /*color=*/0),
  };
  Shape shape = ShapeUtil::MakeShape(S32, {kCount});
  ShapedSlice src_slice{BufferAllocation::Slice(&allocs[0], 0, kBytes), shape};
  ShapedSlice dst_slice{BufferAllocation::Slice(&allocs[1], 0, kBytes), shape};

  gpu::DeviceToDeviceCopyThunk thunk(gpu::Thunk::ThunkInfo(), src_slice,
                                     dst_slice, kBytes);

  se::StreamExecutorAddressAllocator allocator(executor);
  gpu::BufferAllocations buffer_allocations({src_dev, dst_dev},
                                             /*device_ordinal=*/0, &allocator);

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = gpu::Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));

  std::vector<int32_t> readback(kCount, 0);
  TF_ASSERT_OK(stream->Memcpy(readback.data(), dst_dev, kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(readback, pattern);

  executor->Deallocate(&src_dev);
  executor->Deallocate(&dst_dev);
}

// WhileThunk with trip_count runs the body N times, ignoring the condition
// computation path. Body is a KernelThunk that increments a counter; after
// 5 iterations the counter should read 5.
TEST(MetalKernelThunkTest, WhileThunkTripCount) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  se::DeviceAddress<int32_t> counter_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  ASSERT_NE(counter_dev.opaque(), nullptr);
  int32_t zero = 0;
  TF_ASSERT_OK(stream->Memcpy(&counter_dev, &zero, sizeof(int32_t)));

  // One real BufferAllocation for the counter; a dummy 1-byte allocation for
  // the unused condition-result slot (trip_count short-circuits the read).
  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1, /*color=*/0),
  };
  emitters::KernelArgument arg_counter(ShapeUtil::MakeShape(S32, {1}),
                                       BufferAllocation::Slice(&allocs[0], 0,
                                                               4));
  arg_counter.set_written(true);

  auto body_kernel = std::make_unique<gpu::KernelThunk>(
      gpu::Thunk::ThunkInfo(), /*kernel_name=*/"increment",
      emitters::KernelArguments({arg_counter}),
      gpu::LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  gpu::ThunkSequence body_thunks;
  body_thunks.push_back(std::move(body_kernel));

  // condition_result_buffer_index points at the dummy 1-byte allocation; not
  // read because trip_count_ is set.
  BufferAllocation::Slice cond_slice(&allocs[1], 0, 1);
  gpu::WhileThunk while_thunk(gpu::Thunk::ThunkInfo(), cond_slice,
                              /*condition_thunks=*/gpu::ThunkSequence(),
                              /*body_thunks=*/std::move(body_thunks),
                              /*trip_count=*/5);

  // We need a dummy device buffer for the condition slot too — the
  // BufferAllocations array maps allocation index to address.
  se::DeviceAddress<int8_t> cond_dev =
      executor->AllocateArray<int8_t>(1, /*memory_space=*/0);

  se::StreamExecutorAddressAllocator allocator(executor);
  gpu::BufferAllocations buffer_allocations({counter_dev, cond_dev},
                                             /*device_ordinal=*/0, &allocator);

  gpu::Thunk::ExecutableSource src;
  src.text = kIncrementSource;

  gpu::Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.src = src;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &buffer_allocations;
  TF_ASSERT_OK(while_thunk.Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = gpu::Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});
  TF_ASSERT_OK(while_thunk.ExecuteOnStream(execute_params));

  int32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, counter_dev, sizeof(int32_t)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 5);

  executor->Deallocate(&counter_dev);
  executor->Deallocate(&cond_dev);
}

// GpuExecutable::Create with a real ThunkSequence containing our add_i32
// KernelThunk should succeed for Metal — proving that the executable shell
// is reusable too, not just the thunks underneath it.
TEST(MetalGpuExecutableTest, CreateAcceptsMetalThunks) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  // Build the same add_i32 KernelThunk we validated standalone.
  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  auto kernel_thunk = std::make_unique<gpu::KernelThunk>(
      gpu::Thunk::ThunkInfo(),
      /*kernel_name=*/"add_i32",
      emitters::KernelArguments({arg_a, arg_b, arg_c}),
      gpu::LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  gpu::ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk));

  gpu::GpuExecutable::Params params;
  params.executable =
      std::make_unique<gpu::ThunkExecutor>(std::move(thunk_sequence));
  params.mlir_allocations = std::move(allocs);
  params.device_description = executor->GetDeviceDescription();
  params.module_name = "metal_executable_smoke";
  params.enable_debug_info_manager = false;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<gpu::GpuExecutable> executable,
                          gpu::GpuExecutable::Create(std::move(params)));
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->name(), "metal_executable_smoke");
}

// Build a GpuExecutable around the add_i32 kernel and actually execute it
// end to end through ExecuteAsyncOnStream. Inputs come in as ShapedBuffers
// (parameters); the output buffer is allocated by the runtime via our
// StreamExecutorAddressAllocator -> MetalAllocator path and returned in a
// ScopedShapedBuffer. We read back from its root buffer and verify
// c == a + b.
TEST(MetalGpuExecutableTest, ExecuteAsyncOnStream) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  // Inputs: parameter 0 (a) and parameter 1 (b); output: maybe_live_out (c).
  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  allocs[0].set_entry_computation_parameter(/*parameter_number=*/0,
                                            /*param_shape_index=*/{},
                                            /*parameter_aliased_with_output=*/
                                            false);
  allocs[1].set_entry_computation_parameter(1, {}, false);
  allocs[2].set_maybe_live_out(true);

  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  auto kernel_thunk = std::make_unique<gpu::KernelThunk>(
      gpu::Thunk::ThunkInfo(),
      /*kernel_name=*/"add_i32",
      emitters::KernelArguments({arg_a, arg_b, arg_c}),
      gpu::LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  gpu::ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk));

  ProgramShape program_shape;
  program_shape.AddParameter(ShapeUtil::MakeShape(S32, {1}), "a");
  program_shape.AddParameter(ShapeUtil::MakeShape(S32, {1}), "b");
  *program_shape.mutable_result() = ShapeUtil::MakeShape(S32, {1});

  gpu::GpuExecutable::Params params;
  params.executable =
      std::make_unique<gpu::ThunkExecutor>(std::move(thunk_sequence));
  params.mlir_allocations = std::move(allocs);
  params.device_description = executor->GetDeviceDescription();
  params.module_name = "metal_add_i32";
  params.program_shape = std::move(program_shape);
  params.output_info[ShapeIndex{}] = gpu::GpuExecutable::OutputInfo{
      /*allocation_index=*/2,
      /*passthrough=*/false,
      /*alias_config=*/std::nullopt,
  };
  // KernelThunk::Initialize reads InitializeParams::src.text — populated by
  // GpuExecutable from its text_ field, which we set here via asm_text.
  // The name is "asm_text" for historical reasons (PTX on CUDA); for us it
  // is just the MSL source string passed through unchanged.
  params.asm_text = std::string(kAddI32Source);
  params.enable_debug_info_manager = false;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<gpu::GpuExecutable> executable,
                          gpu::GpuExecutable::Create(std::move(params)));

  // Input buffers. Allocate via MetalExecutor; init values; wrap each in a
  // ShapedBuffer that GpuExecutable will read as a parameter.
  se::DeviceAddress<int32_t> a_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  se::DeviceAddress<int32_t> b_dev =
      executor->AllocateArray<int32_t>(1, /*memory_space=*/0);
  ASSERT_NE(a_dev.opaque(), nullptr);
  ASSERT_NE(b_dev.opaque(), nullptr);
  int32_t val_a = 1, val_b = 2;
  TF_ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  ShapedBuffer a_buf(ShapeUtil::MakeShape(S32, {1}), /*device_ordinal=*/0);
  a_buf.set_buffer(a_dev, /*index=*/{});
  ShapedBuffer b_buf(ShapeUtil::MakeShape(S32, {1}), 0);
  b_buf.set_buffer(b_dev, {});

  // Allocator for the output. StreamExecutorAddressAllocator forwards to
  // MetalExecutor::Allocate / MetalAllocator.
  se::StreamExecutorAddressAllocator allocator(executor);

  ExecutableRunOptions run_options;
  run_options.set_stream(stream.get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(0);
  ServiceExecutableRunOptions service_run_options(run_options);

  std::vector<const ShapedBuffer*> arg_buffers = {&a_buf, &b_buf};
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      executable->ExecuteAsyncOnStream(&service_run_options, arg_buffers));

  int32_t output = 0;
  TF_ASSERT_OK(
      stream->Memcpy(&output, result.root_buffer(), sizeof(int32_t)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(output, 3);

  executor->Deallocate(&a_dev);
  executor->Deallocate(&b_dev);
}

}  // namespace
}  // namespace metal
}  // namespace xla
