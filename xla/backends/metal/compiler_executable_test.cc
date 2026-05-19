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

// End-to-end validation of MetalCompiler::RunBackend. Wedge H bound: the
// only HLO shape it accepts is one whose entry root is a parameter
// (passthrough). Subsequent wedges add real kernel emission for compute
// ops.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/metal/compiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
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

constexpr absl::string_view kPassthroughHlo = R"hlo(
HloModule passthrough
ENTRY main {
  ROOT p = f32[4] parameter(0)
}
)hlo";

TEST(MetalCompilerExecutableTest, RunBackendBuildsPassthroughExecutable) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kPassthroughHlo));

  MetalCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(module), executor, Compiler::CompileOptions{}));
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->module().name(), "passthrough");
}

TEST(MetalCompilerExecutableTest, RunBackendLowersAddViaFusionPath) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  // An add HLO is wrapped in a Loop fusion by GpuCompiler's FusionWrapper
  // pass during ScheduleAndVerify. The walker's kFusion case then routes
  // it through MlirKernelEmitter → metal::EmitMslKernel → gpu::KernelThunk.
  // We only verify the compile succeeds and produces an executable —
  // execution coverage for non-passthrough HLO lives in a separate test.
  constexpr absl::string_view kAddHlo = R"hlo(
    HloModule add
    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT r = f32[4] add(a, b)
    }
  )hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kAddHlo));

  MetalCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(module), executor,
                          Compiler::CompileOptions{}));
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->module().name(), "add");
}

TEST(MetalCompilerExecutableTest, ExecutePassthroughReturnsInputBuffer) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kPassthroughHlo));

  MetalCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(module), executor,
                          Compiler::CompileOptions{}));

  constexpr int kCount = 4;
  constexpr int64_t kBytes = kCount * sizeof(float);
  se::DeviceAddress<float> input_dev =
      executor->AllocateArray<float>(kCount, /*memory_space=*/0);
  ASSERT_NE(input_dev.opaque(), nullptr);
  std::vector<float> pattern{1.5f, -2.0f, 3.25f, 4.75f};
  TF_ASSERT_OK(stream->Memcpy(&input_dev, pattern.data(), kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  ShapedBuffer input_buf(ShapeUtil::MakeShape(F32, {kCount}),
                         /*device_ordinal=*/0);
  input_buf.set_buffer(input_dev, /*index=*/{});

  se::StreamExecutorAddressAllocator allocator(executor);
  ExecutableRunOptions run_options;
  run_options.set_stream(stream.get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(0);
  ServiceExecutableRunOptions service_run_options(run_options);

  std::vector<const ShapedBuffer*> args = {&input_buf};
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      executable->ExecuteAsyncOnStream(&service_run_options, args));

  std::vector<float> readback(kCount, 0.0f);
  TF_ASSERT_OK(stream->Memcpy(readback.data(), result.root_buffer(), kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(readback, pattern);

  executor->Deallocate(&input_dev);
}

TEST(MetalCompilerExecutableTest, ExecuteAddProducesElementwiseSum) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  constexpr absl::string_view kAddHlo = R"hlo(
    HloModule add
    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT r = f32[4] add(a, b)
    }
  )hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kAddHlo));

  MetalCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(module), executor,
                          Compiler::CompileOptions{}));

  constexpr int kCount = 4;
  constexpr int64_t kBytes = kCount * sizeof(float);
  std::vector<float> pattern_a{1.5f, -2.0f, 3.25f, 4.75f};
  std::vector<float> pattern_b{0.5f, 10.0f, -0.25f, 100.0f};

  se::DeviceAddress<float> a_dev =
      executor->AllocateArray<float>(kCount, /*memory_space=*/0);
  ASSERT_NE(a_dev.opaque(), nullptr);
  se::DeviceAddress<float> b_dev =
      executor->AllocateArray<float>(kCount, /*memory_space=*/0);
  ASSERT_NE(b_dev.opaque(), nullptr);
  TF_ASSERT_OK(stream->Memcpy(&a_dev, pattern_a.data(), kBytes));
  TF_ASSERT_OK(stream->Memcpy(&b_dev, pattern_b.data(), kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  ShapedBuffer a_buf(ShapeUtil::MakeShape(F32, {kCount}), /*device_ordinal=*/0);
  a_buf.set_buffer(a_dev, /*index=*/{});
  ShapedBuffer b_buf(ShapeUtil::MakeShape(F32, {kCount}), /*device_ordinal=*/0);
  b_buf.set_buffer(b_dev, /*index=*/{});

  se::StreamExecutorAddressAllocator allocator(executor);
  ExecutableRunOptions run_options;
  run_options.set_stream(stream.get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(0);
  ServiceExecutableRunOptions service_run_options(run_options);

  std::vector<const ShapedBuffer*> args = {&a_buf, &b_buf};
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      executable->ExecuteAsyncOnStream(&service_run_options, args));

  std::vector<float> readback(kCount, 0.0f);
  TF_ASSERT_OK(stream->Memcpy(readback.data(), result.root_buffer(), kBytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<float> expected(kCount);
  for (int i = 0; i < kCount; ++i) {
    expected[i] = pattern_a[i] + pattern_b[i];
  }
  EXPECT_EQ(readback, expected);

  executor->Deallocate(&a_dev);
  executor->Deallocate(&b_dev);
}

}  // namespace
}  // namespace metal
}  // namespace xla
