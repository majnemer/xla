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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_DEVICE_HANDLE_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_DEVICE_HANDLE_H_

namespace stream_executor {

class StreamExecutor;

namespace metal {

// Returns the underlying id<MTLDevice> as a void* (toll-free bridged
// to/from id<MTLDevice> via __bridge inside .mm consumers). Used by Metal-
// targeting compiler-side code (xla/backends/metal/*.cc) that needs the
// device handle but cannot include the Obj-C executor header.
//
// Returns nullptr if `stream_exec` is not a MetalExecutor or hasn't been
// Init()'d. The returned pointer is owned by the executor; the caller must
// not release it. It remains valid for the executor's lifetime.
void* GetMetalDeviceOpaque(StreamExecutor* stream_exec);

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_DEVICE_HANDLE_H_
