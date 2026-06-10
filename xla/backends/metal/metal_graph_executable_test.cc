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

// End-to-end coverage for __metal_graph regions: RunHloPasses forms the
// fusions (MetalGraphPartitioner), RunBackend compiles MPSGraphExecutables,
// and execution flows through MetalGraphThunk. Numeric results are compared
// against host references.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/metal/compiler.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
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

int CountGraphFusions(const HloModule& module) {
  int count = 0;
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    count += IsMetalGraphFusion(*instr) ? 1 : 0;
  }
  return count;
}

// Compiles via the full pipeline (RunHloPasses + RunBackend), asserting the
// optimized module contains exactly `expected_graph_fusions` __metal_graph
// fusions in the entry computation.
std::unique_ptr<Executable> CompileOrFail(se::StreamExecutor* executor,
                                          absl::string_view hlo,
                                          int expected_graph_fusions) {
  auto parsed = ParseAndReturnUnverifiedModule(hlo);
  if (!parsed.ok()) {
    ADD_FAILURE() << "parse failed: " << parsed.status();
    return nullptr;
  }
  MetalCompiler compiler;
  auto optimized = compiler.RunHloPasses(*std::move(parsed), executor,
                                         Compiler::CompileOptions{});
  if (!optimized.ok()) {
    ADD_FAILURE() << "RunHloPasses failed: " << optimized.status();
    return nullptr;
  }
  EXPECT_EQ(CountGraphFusions(**optimized), expected_graph_fusions)
      << (*optimized)->ToString();
  auto executable = compiler.RunBackend(*std::move(optimized), executor,
                                        Compiler::CompileOptions{});
  if (!executable.ok()) {
    ADD_FAILURE() << "RunBackend failed: " << executable.status();
    return nullptr;
  }
  return *std::move(executable);
}

// Uploads f32 inputs, executes, and returns the f32 readback of the result.
std::vector<float> ExecuteF32(se::StreamExecutor* executor,
                              Executable* executable,
                              const std::vector<Shape>& input_shapes,
                              const std::vector<std::vector<float>>& inputs,
                              int64_t result_count) {
  auto stream = executor->CreateStream();
  TF_EXPECT_OK(stream.status());

  std::vector<se::DeviceAddress<float>> device_inputs;
  std::vector<ShapedBuffer> buffers;
  buffers.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    se::DeviceAddress<float> dev =
        executor->AllocateArray<float>(inputs[i].size(), /*memory_space=*/0);
    EXPECT_NE(dev.opaque(), nullptr);
    TF_EXPECT_OK((*stream)->Memcpy(&dev, inputs[i].data(),
                                   inputs[i].size() * sizeof(float)));
    device_inputs.push_back(dev);
    buffers.emplace_back(input_shapes[i], /*device_ordinal=*/0);
    buffers.back().set_buffer(dev, /*index=*/{});
  }
  TF_EXPECT_OK((*stream)->BlockHostUntilDone());

  se::StreamExecutorAddressAllocator allocator(executor);
  ExecutableRunOptions run_options;
  run_options.set_stream(stream->get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(0);
  ServiceExecutableRunOptions service_run_options(run_options);

  std::vector<const ShapedBuffer*> args;
  args.reserve(buffers.size());
  for (const ShapedBuffer& buffer : buffers) {
    args.push_back(&buffer);
  }
  auto result =
      executable->ExecuteAsyncOnStream(&service_run_options, args);
  TF_EXPECT_OK(result.status());

  std::vector<float> readback(result_count, 0.0f);
  TF_EXPECT_OK((*stream)->Memcpy(readback.data(), result->root_buffer(),
                                 result_count * sizeof(float)));
  TF_EXPECT_OK((*stream)->BlockHostUntilDone());

  for (se::DeviceAddress<float>& dev : device_inputs) {
    executor->Deallocate(&dev);
  }
  return readback;
}

void ExpectNear(const std::vector<float>& actual,
                const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
  }
}

TEST(MetalGraphExecutableTest, SoloDotMatchesHostReference) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  constexpr absl::string_view kHlo = R"hlo(
    HloModule dot
    ENTRY main {
      a = f32[4,8] parameter(0)
      b = f32[8,3] parameter(1)
      ROOT d = f32[4,3] dot(a, b), lhs_contracting_dims={1},
                                   rhs_contracting_dims={0}
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/1);
  ASSERT_NE(executable, nullptr);

  std::vector<float> a(4 * 8), b(8 * 3);
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.25f * i - 3.0f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = 0.125f * i + 0.5f;
  std::vector<float> expected(4 * 3, 0.0f);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      for (int k = 0; k < 8; ++k) {
        expected[m * 3 + n] += a[m * 8 + k] * b[k * 3 + n];
      }
    }
  }
  std::vector<float> actual = ExecuteF32(
      executor, executable.get(),
      {ShapeUtil::MakeShape(F32, {4, 8}), ShapeUtil::MakeShape(F32, {8, 3})},
      {a, b}, expected.size());
  ExpectNear(actual, expected, 1e-4f);
}

TEST(MetalGraphExecutableTest, DotTanhDotIsOneRegion) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  constexpr absl::string_view kHlo = R"hlo(
    HloModule dot_tanh_dot
    ENTRY main {
      a = f32[4,4] parameter(0)
      b = f32[4,4] parameter(1)
      c = f32[4,4] parameter(2)
      d0 = f32[4,4] dot(a, b), lhs_contracting_dims={1},
                               rhs_contracting_dims={0}
      t = f32[4,4] tanh(d0)
      ROOT d1 = f32[4,4] dot(t, c), lhs_contracting_dims={1},
                                    rhs_contracting_dims={0}
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/1);
  ASSERT_NE(executable, nullptr);

  std::vector<float> a(16), b(16), c(16);
  for (int i = 0; i < 16; ++i) {
    a[i] = 0.1f * i - 0.8f;
    b[i] = 0.05f * i + 0.1f;
    c[i] = 0.2f * ((i * 7) % 5) - 0.4f;
  }
  auto matmul4 = [](const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float> out(16, 0.0f);
    for (int m = 0; m < 4; ++m) {
      for (int n = 0; n < 4; ++n) {
        for (int k = 0; k < 4; ++k) {
          out[m * 4 + n] += x[m * 4 + k] * y[k * 4 + n];
        }
      }
    }
    return out;
  };
  std::vector<float> inner = matmul4(a, b);
  for (float& v : inner) v = std::tanh(v);
  std::vector<float> expected = matmul4(inner, c);

  Shape s = ShapeUtil::MakeShape(F32, {4, 4});
  std::vector<float> actual =
      ExecuteF32(executor, executable.get(), {s, s, s}, {a, b, c}, 16);
  ExpectNear(actual, expected, 1e-3f);
}

TEST(MetalGraphExecutableTest, DotBiasActivationEpilogue) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  constexpr absl::string_view kHlo = R"hlo(
    HloModule dot_bias
    ENTRY main {
      a = f32[4,8] parameter(0)
      b = f32[8,3] parameter(1)
      bias = f32[3] parameter(2)
      d = f32[4,3] dot(a, b), lhs_contracting_dims={1},
                              rhs_contracting_dims={0}
      bc = f32[4,3] broadcast(bias), dimensions={1}
      add = f32[4,3] add(d, bc)
      ROOT t = f32[4,3] tanh(add)
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/1);
  ASSERT_NE(executable, nullptr);

  std::vector<float> a(4 * 8), b(8 * 3), bias{0.5f, -1.0f, 2.0f};
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.05f * i - 1.0f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = 0.07f * i - 0.7f;
  std::vector<float> expected(4 * 3, 0.0f);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      float acc = bias[n];
      for (int k = 0; k < 8; ++k) {
        acc += a[m * 8 + k] * b[k * 3 + n];
      }
      expected[m * 3 + n] = std::tanh(acc);
    }
  }
  std::vector<float> actual = ExecuteF32(
      executor, executable.get(),
      {ShapeUtil::MakeShape(F32, {4, 8}), ShapeUtil::MakeShape(F32, {8, 3}),
       ShapeUtil::MakeShape(F32, {3})},
      {a, b, bias}, expected.size());
  ExpectNear(actual, expected, 1e-4f);
}

TEST(MetalGraphExecutableTest, DotExpReduceCapturesThroughMonoid) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  constexpr absl::string_view kHlo = R"hlo(
    HloModule dot_reduce
    add_comp {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT s = f32[] add(x, y)
    }
    ENTRY main {
      a = f32[4,8] parameter(0)
      b = f32[8,3] parameter(1)
      d = f32[4,3] dot(a, b), lhs_contracting_dims={1},
                              rhs_contracting_dims={0}
      e = f32[4,3] exponential(d)
      c0 = f32[] constant(0)
      ROOT r = f32[4] reduce(e, c0), dimensions={1}, to_apply=add_comp
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/1);
  ASSERT_NE(executable, nullptr);

  std::vector<float> a(4 * 8), b(8 * 3);
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.02f * i - 0.3f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = 0.03f * i - 0.35f;
  std::vector<float> expected(4, 0.0f);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < 8; ++k) {
        acc += a[m * 8 + k] * b[k * 3 + n];
      }
      expected[m] += std::exp(acc);
    }
  }
  std::vector<float> actual = ExecuteF32(
      executor, executable.get(),
      {ShapeUtil::MakeShape(F32, {4, 8}), ShapeUtil::MakeShape(F32, {8, 3})},
      {a, b}, expected.size());
  ExpectNear(actual, expected, 1e-3f);
}

TEST(MetalGraphExecutableTest, Conv2DMatchesHostReference) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  // NHWC input, HWIO kernel — valid padding, stride 1.
  constexpr absl::string_view kHlo = R"hlo(
    HloModule conv
    ENTRY main {
      input = f32[1,4,4,2] parameter(0)
      kernel = f32[2,2,2,3] parameter(1)
      ROOT conv = f32[1,3,3,3] convolution(input, kernel),
          window={size=2x2}, dim_labels=b01f_01io->b01f
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/1);
  ASSERT_NE(executable, nullptr);

  std::vector<float> input(1 * 4 * 4 * 2), kernel(2 * 2 * 2 * 3);
  for (size_t i = 0; i < input.size(); ++i) input[i] = 0.1f * i - 1.5f;
  for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = 0.05f * i - 0.5f;

  auto in_at = [&](int h, int w, int c) {
    return input[(h * 4 + w) * 2 + c];
  };
  auto k_at = [&](int kh, int kw, int ci, int co) {
    return kernel[((kh * 2 + kw) * 2 + ci) * 3 + co];
  };
  std::vector<float> expected(3 * 3 * 3, 0.0f);
  for (int h = 0; h < 3; ++h) {
    for (int w = 0; w < 3; ++w) {
      for (int co = 0; co < 3; ++co) {
        float acc = 0.0f;
        for (int kh = 0; kh < 2; ++kh) {
          for (int kw = 0; kw < 2; ++kw) {
            for (int ci = 0; ci < 2; ++ci) {
              acc += in_at(h + kh, w + kw, ci) * k_at(kh, kw, ci, co);
            }
          }
        }
        expected[(h * 3 + w) * 3 + co] = acc;
      }
    }
  }
  std::vector<float> actual = ExecuteF32(
      executor, executable.get(),
      {ShapeUtil::MakeShape(F32, {1, 4, 4, 2}),
       ShapeUtil::MakeShape(F32, {2, 2, 2, 3})},
      {input, kernel}, expected.size());
  ExpectNear(actual, expected, 1e-3f);
}

TEST(MetalGraphExecutableTest, IntegerDotFallsBackToElementalMsl) {
  se::StreamExecutor* executor = GetMetalExecutorOrFail();
  ASSERT_NE(executor, nullptr);

  // s32 dots are gate-refused: no __metal_graph fusion forms, and the dot
  // flows through the solo-fusion elemental MSL path unchanged.
  constexpr absl::string_view kHlo = R"hlo(
    HloModule int_dot
    ENTRY main {
      a = s32[4,8] parameter(0)
      b = s32[8,3] parameter(1)
      ROOT d = s32[4,3] dot(a, b), lhs_contracting_dims={1},
                                   rhs_contracting_dims={0}
    }
  )hlo";
  std::unique_ptr<Executable> executable =
      CompileOrFail(executor, kHlo, /*expected_graph_fusions=*/0);
  ASSERT_NE(executable, nullptr);

  auto stream = executor->CreateStream();
  TF_ASSERT_OK(stream.status());
  std::vector<int32_t> a(4 * 8), b(8 * 3);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int32_t>(i) - 13;
  for (size_t i = 0; i < b.size(); ++i) b[i] = 2 * static_cast<int32_t>(i) - 19;
  std::vector<int32_t> expected(4 * 3, 0);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      for (int k = 0; k < 8; ++k) {
        expected[m * 3 + n] += a[m * 8 + k] * b[k * 3 + n];
      }
    }
  }

  se::DeviceAddress<int32_t> a_dev =
      executor->AllocateArray<int32_t>(a.size(), 0);
  se::DeviceAddress<int32_t> b_dev =
      executor->AllocateArray<int32_t>(b.size(), 0);
  TF_ASSERT_OK((*stream)->Memcpy(&a_dev, a.data(), a.size() * sizeof(int32_t)));
  TF_ASSERT_OK((*stream)->Memcpy(&b_dev, b.data(), b.size() * sizeof(int32_t)));
  TF_ASSERT_OK((*stream)->BlockHostUntilDone());

  ShapedBuffer a_buf(ShapeUtil::MakeShape(S32, {4, 8}), 0);
  a_buf.set_buffer(a_dev, {});
  ShapedBuffer b_buf(ShapeUtil::MakeShape(S32, {8, 3}), 0);
  b_buf.set_buffer(b_dev, {});

  se::StreamExecutorAddressAllocator allocator(executor);
  ExecutableRunOptions run_options;
  run_options.set_stream(stream->get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(0);
  ServiceExecutableRunOptions service_run_options(run_options);

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      executable->ExecuteAsyncOnStream(&service_run_options,
                                       {&a_buf, &b_buf}));
  std::vector<int32_t> readback(expected.size(), 0);
  TF_ASSERT_OK((*stream)->Memcpy(readback.data(), result.root_buffer(),
                                 readback.size() * sizeof(int32_t)));
  TF_ASSERT_OK((*stream)->BlockHostUntilDone());
  EXPECT_EQ(readback, expected);

  executor->Deallocate(&a_dev);
  executor->Deallocate(&b_dev);
}

}  // namespace
}  // namespace metal
}  // namespace xla
