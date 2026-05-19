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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_ALLOCATOR_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <map>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor {
namespace metal {

// Per-executor allocator that owns MTLBuffer instances and exposes them as
// raw byte addresses.
//
// All buffers are allocated with MTLResourceStorageModeShared. On Apple
// Silicon (which is the only family we target — see metal_executor.mm) this
// is unified memory: the buffer's contents() pointer is also a valid CPU
// pointer to the same backing storage, so DeviceAddressBase pointer
// arithmetic (e.g. GetByteSlice) yields a pointer that the host can both
// read/write and resolve back to an (MTLBuffer, offset) pair for blit
// encoders and kernel binding.
//
// Buffers are indexed by their base contents pointer in a sorted map; Resolve
// uses std::upper_bound to find the buffer that contains an arbitrary
// in-range pointer. Allocate / Deallocate / Resolve are thread-safe.
class MetalAllocator {
 public:
  explicit MetalAllocator(id<MTLDevice> device);

  // Allocates `size` bytes and returns the base contents pointer of the
  // backing MTLBuffer. Returns ResourceExhausted on Metal allocation
  // failure. Size 0 returns nullptr (no allocation).
  absl::StatusOr<void*> Allocate(uint64_t size);

  // Releases the MTLBuffer whose base contents pointer is `base`. If `base`
  // is nullptr or not owned by this allocator, this is a no-op (matches the
  // free(nullptr) tradition used by other backends).
  void Deallocate(void* base);

  struct Resolved {
    id<MTLBuffer> buffer;
    uint64_t offset;
  };

  // Given any byte pointer `ptr` that lies within one of our allocations,
  // returns the (MTLBuffer, offset) pair. Returns nullopt if `ptr` is not
  // within any owned allocation.
  std::optional<Resolved> Resolve(const void* ptr) const;

 private:
  id<MTLDevice> device_;

  mutable absl::Mutex mu_;
  // base contents pointer -> MTLBuffer. Sorted by key so Resolve can use
  // upper_bound to find the buffer whose range contains a probe pointer.
  std::map<void*, id<MTLBuffer>> buffers_ ABSL_GUARDED_BY(mu_);

  MetalAllocator(const MetalAllocator&) = delete;
  MetalAllocator& operator=(const MetalAllocator&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_ALLOCATOR_H_
