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
#include "xla/stream_executor/gpu/repeat_buffer_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/metal/metal_platform_id.h"

namespace stream_executor::metal {

// MSL mirror of RepeatBufferKernelImpl (repeat_buffer_kernel.cu.h): populate
// the last `buffer_size - repeat_size` bytes of `buffer` by repeating the
// first `repeat_size` bytes. Launched with at least `repeat_size` threads.
constexpr const char kRepeatBufferKernelMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void repeat_buffer_kernel(device char* buffer [[buffer(0)]],
                                 constant long& repeat_size [[buffer(1)]],
                                 constant long& buffer_size [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
  const long global_index = (long)gid;
  if (global_index >= repeat_size) {
    return;
  }
  const char src_value = buffer[global_index];
  for (long dst_index = global_index + repeat_size; dst_index < buffer_size;
       dst_index += repeat_size) {
    buffer[dst_index] = src_value;
  }
}
)msl";

}  // namespace stream_executor::metal

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    RepeatBufferKernelMetal, stream_executor::gpu::RepeatBufferKernel,
    stream_executor::metal::kMetalPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateMslSourceInMemorySpec(
          stream_executor::metal::kRepeatBufferKernelMsl,
          "repeat_buffer_kernel", arity);
    }));
