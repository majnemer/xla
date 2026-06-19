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

#include "xla/stream_executor/metal/metal_allocator.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <map>
#include <optional>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/memory_space.h"

namespace stream_executor {
namespace metal {

MetalAllocator::MetalAllocator(id<MTLDevice> device) : device_(device) {}

absl::StatusOr<void *> MetalAllocator::Allocate(uint64_t size,
                                                MemorySpace memory_space) {
  @autoreleasepool {
    if (size == 0) {
      return nullptr;
    }
    id<MTLBuffer> buffer =
        [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return absl::ResourceExhaustedError(
          absl::StrCat("MetalAllocator::Allocate: [newBufferWithLength:", size,
                       "] returned nil."));
    }
    void *base = [buffer contents];
    if (base == nullptr) {
      return absl::InternalError(
          "MetalAllocator::Allocate: MTLBuffer.contents was nullptr; "
          "Shared-mode buffer should always expose a CPU pointer.");
    }
    absl::MutexLock lock(&mu_);
    buffers_.emplace(base, Allocation{buffer, memory_space});
    return base;
  }
}

void MetalAllocator::Deallocate(void *base) {
  if (base == nullptr) {
    return;
  }
  // Erasing the map entry drops the last reference to the MTLBuffer and runs
  // its dealloc, which autoreleases the buffer's device reference. Bound that
  // here so it does not leak when Deallocate runs on a thread with no pool.
  @autoreleasepool {
    absl::MutexLock lock(&mu_);
    buffers_.erase(base);
  }
}

std::optional<MetalAllocator::Resolved>
MetalAllocator::Resolve(const void *ptr) const {
  @autoreleasepool {
    if (ptr == nullptr) {
      return std::nullopt;
    }
    absl::MutexLock lock(&mu_);
    // upper_bound returns the first entry whose key is strictly greater than
    // ptr; the predecessor is the candidate that may contain ptr.
    auto it = buffers_.upper_bound(const_cast<void *>(ptr));
    if (it == buffers_.begin()) {
      return std::nullopt;
    }
    --it;
    void *base = it->first;
    id<MTLBuffer> buffer = it->second.buffer;
    uint64_t length = static_cast<uint64_t>([buffer length]);
    auto offset = static_cast<uint64_t>(reinterpret_cast<const char *>(ptr) -
                                        reinterpret_cast<char *>(base));
    if (offset >= length) {
      return std::nullopt;
    }
    return Resolved{buffer, offset, it->second.memory_space};
  }
}

} // namespace metal
} // namespace stream_executor
