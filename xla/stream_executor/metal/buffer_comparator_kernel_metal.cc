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

#include <cstdint>
#include <deque>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/stream_executor/gpu/buffer_comparator_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/types.h"

namespace stream_executor::metal {
namespace {

// MSL mirrors of the comparison kernels in buffer_comparator_kernel_lib
// .cu.h: grid-stride loops counting elements whose relative error
// |a-b| / (max(|a|,|b|) + 1) exceeds the threshold. NaN == NaN; half
// infinities clamp to +/-65504 before comparing; integers compute the exact
// integer distance before converting to float for the tolerance calculation.
// The 32-bit add matches CUDA; the host reads the count from a
// zero-initialized 8-byte buffer.
//
// $0 = kernel name, $1 = MSL element type, $2/$3 = elem_a/elem_b loads.
constexpr absl::string_view kFloatComparisonMslTemplate = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void $0(device const $1* buffer_a [[buffer(0)]],
               device const $1* buffer_b [[buffer(1)]],
               constant float& rel_error_threshold [[buffer(2)]],
               constant ulong& buffer_length [[buffer(3)]],
               device atomic_uint* mismatch_count [[buffer(4)]],
               uint gid [[thread_position_in_grid]],
               uint grid_size [[threads_per_grid]]) {
  for (ulong idx = gid; idx < buffer_length; idx += grid_size) {
    float elem_a = $2;
    float elem_b = $3;
    if (isnan(elem_a) && isnan(elem_b)) {
      continue;
    }
    if (elem_a == elem_b) {
      continue;
    }
    float rel_error =
        abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1.0f);
    if (rel_error > rel_error_threshold || isnan(rel_error)) {
      atomic_fetch_add_explicit(mismatch_count, 1u, memory_order_relaxed);
    }
  }
}
)msl";

constexpr absl::string_view kSignedIntComparisonMslTemplate = R"msl(
#include <metal_stdlib>
using namespace metal;

static inline bool compare_equal(long lhs, long rhs,
                                 float rel_error_threshold) {
  if ((lhs < 0) != (rhs < 0)) {
    return false;
  }
  long max_elem = lhs < rhs ? rhs : lhs;
  long min_elem = lhs < rhs ? lhs : rhs;
  ulong abs_diff = (ulong)(max_elem - min_elem);
  ulong max_abs;
  if (max_elem < 0) {
    max_abs = 0ul - (ulong)min_elem;
  } else {
    max_abs = (ulong)max_elem;
  }
  float rel_error = (float)abs_diff / ((float)max_abs + 1.0f);
  return rel_error <= rel_error_threshold;
}

kernel void $0(device const $1* buffer_a [[buffer(0)]],
               device const $1* buffer_b [[buffer(1)]],
               constant float& rel_error_threshold [[buffer(2)]],
               constant ulong& buffer_length [[buffer(3)]],
               device atomic_uint* mismatch_count [[buffer(4)]],
               uint gid [[thread_position_in_grid]],
               uint grid_size [[threads_per_grid]]) {
  for (ulong idx = gid; idx < buffer_length; idx += grid_size) {
    long elem_a = (long)buffer_a[idx];
    long elem_b = (long)buffer_b[idx];
    if (!compare_equal(elem_a, elem_b, rel_error_threshold)) {
      atomic_fetch_add_explicit(mismatch_count, 1u, memory_order_relaxed);
    }
  }
}
)msl";

constexpr absl::string_view kUnsignedIntComparisonMslTemplate = R"msl(
#include <metal_stdlib>
using namespace metal;

static inline bool compare_equal(ulong lhs, ulong rhs,
                                 float rel_error_threshold) {
  ulong max_elem = lhs < rhs ? rhs : lhs;
  ulong min_elem = lhs < rhs ? lhs : rhs;
  ulong abs_diff = max_elem - min_elem;
  ulong max_abs = max_elem;
  float rel_error = (float)abs_diff / ((float)max_abs + 1.0f);
  return rel_error <= rel_error_threshold;
}

kernel void $0(device const $1* buffer_a [[buffer(0)]],
               device const $1* buffer_b [[buffer(1)]],
               constant float& rel_error_threshold [[buffer(2)]],
               constant ulong& buffer_length [[buffer(3)]],
               device atomic_uint* mismatch_count [[buffer(4)]],
               uint gid [[thread_position_in_grid]],
               uint grid_size [[threads_per_grid]]) {
  for (ulong idx = gid; idx < buffer_length; idx += grid_size) {
    ulong elem_a = (ulong)buffer_a[idx];
    ulong elem_b = (ulong)buffer_b[idx];
    if (!compare_equal(elem_a, elem_b, rel_error_threshold)) {
      atomic_fetch_add_explicit(mismatch_count, 1u, memory_order_relaxed);
    }
  }
}
)msl";

std::string LoadExpression(absl::string_view buffer, bool is_half,
                           bool is_signed_int, bool is_unsigned_int) {
  if (is_half) {
    // Clamp half infinities to the largest finite half, as float.
    return absl::Substitute(
        "(isinf((float)$0[idx]) ? copysign(65504.0f, (float)$0[idx]) "
        ": (float)$0[idx])",
        buffer);
  }
  if (is_signed_int) {
    return absl::Substitute("(float)((long)$0[idx])", buffer);
  }
  if (is_unsigned_int) {
    return absl::Substitute("(float)((ulong)$0[idx])", buffer);
  }
  return absl::Substitute("$0[idx]", buffer);
}

template <typename NativeT>
void RegisterComparator(absl::string_view name, absl::string_view msl_type,
                        bool is_half, bool is_signed_int,
                        bool is_unsigned_int) {
  // Sources must outlive the process: the loader spec holds a view and the
  // executor's MSL cache keys by source pointer.
  static auto* sources = new std::deque<std::string>();
  if (is_signed_int) {
    sources->push_back(
        absl::Substitute(kSignedIntComparisonMslTemplate, name, msl_type));
  } else if (is_unsigned_int) {
    sources->push_back(
        absl::Substitute(kUnsignedIntComparisonMslTemplate, name, msl_type));
  } else {
    sources->push_back(absl::Substitute(
        kFloatComparisonMslTemplate, name, msl_type,
        LoadExpression("buffer_a", is_half, is_signed_int, is_unsigned_int),
        LoadExpression("buffer_b", is_half, is_signed_int, is_unsigned_int)));
  }
  KernelLoaderSpec spec = KernelLoaderSpec::CreateMslSourceInMemorySpec(
      sources->back(), std::string(name), /*arity=*/5);
  absl::Status result =
      gpu::GpuKernelRegistry::GetGlobalRegistry()
          .RegisterKernel<gpu::BufferComparatorKernel<NativeT>>(
              kMetalPlatformId, spec);
  if (!result.ok()) {
    LOG(FATAL) << "Failed to register Metal buffer comparator kernel " << name
               << ": " << result;
  }
}

void RegisterBufferComparatorKernelMetalImpl() {
  // f64 (no MSL double), bf16, and the f8/f4 families are intentionally
  // unregistered: lookups fail with NOT_FOUND and the autotuner skips
  // correctness checking for that instruction.
  RegisterComparator<float>("f32_comparison", "float", false, false, false);
  RegisterComparator<Eigen::half>("f16_comparison", "half", true, false, false);
  RegisterComparator<int8_t>("s8_comparison", "char", false, true, false);
  RegisterComparator<uint8_t>("u8_comparison", "uchar", false, false, true);
  RegisterComparator<int16_t>("s16_comparison", "short", false, true, false);
  RegisterComparator<uint16_t>("u16_comparison", "ushort", false, false, true);
  RegisterComparator<int32_t>("s32_comparison", "int", false, true, false);
  RegisterComparator<uint32_t>("u32_comparison", "uint", false, false, true);
  RegisterComparator<int64_t>("s64_comparison", "long", false, true, false);
  RegisterComparator<uint64_t>("u64_comparison", "ulong", false, false, true);
}

}  // namespace
}  // namespace stream_executor::metal

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    RegisterBufferComparatorKernelMetal,
    stream_executor::metal::RegisterBufferComparatorKernelMetalImpl());
