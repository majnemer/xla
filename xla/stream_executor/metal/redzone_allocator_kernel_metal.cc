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

#include <cstddef>

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/redzone_allocator_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/metal/metal_platform_id.h"

namespace stream_executor::metal {

// MSL mirror of RedzoneAllocatorKernelImpl (redzone_allocator_kernel_lib
// .cu.h): grid-stride scan of the redzone bytes, counting bytes that no
// longer match the sentinel pattern. The 32-bit add matches the CUDA
// kernel; the host reads the count from a zero-initialized 8-byte buffer.
constexpr const char kRedzoneAllocatorKernelMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void redzone_checker(device const uchar* input_buffer [[buffer(0)]],
                            constant uchar& redzone_pattern [[buffer(1)]],
                            constant ulong& buffer_length [[buffer(2)]],
                            device atomic_uint* out_mismatches [[buffer(3)]],
                            uint gid [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]]) {
  for (ulong idx = gid; idx < buffer_length; idx += grid_size) {
    if (input_buffer[idx] != redzone_pattern) {
      atomic_fetch_add_explicit(out_mismatches, 1u, memory_order_relaxed);
    }
  }
}
)msl";

}  // namespace stream_executor::metal

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    RedzoneAllocatorKernelMetal, stream_executor::gpu::RedzoneAllocatorKernel,
    stream_executor::metal::kMetalPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateMslSourceInMemorySpec(
          stream_executor::metal::kRedzoneAllocatorKernelMsl,
          "redzone_checker", arity);
    }));
