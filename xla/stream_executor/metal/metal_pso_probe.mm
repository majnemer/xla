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

#import <Metal/Metal.h>

#include "xla/stream_executor/metal/metal_pso_probe.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
namespace metal {

namespace {

std::string DescribeError(NSError *err, absl::string_view stage) {
  NSString *msg = err == nil ? @"(no error info)" : [err description];
  return absl::StrCat(stage, ": ", [msg UTF8String]);
}

} // namespace

PipelineStateRef::PipelineStateRef() : opaque_(nullptr) {}

PipelineStateRef::~PipelineStateRef() {
  if (opaque_ != nullptr) {
    // Convert back to a +1 owned ARC reference and let scope drop release it.
    id<MTLComputePipelineState> pso =
        (__bridge_transfer id<MTLComputePipelineState>)opaque_;
    (void)pso;
    opaque_ = nullptr;
  }
}

PipelineStateRef::PipelineStateRef(PipelineStateRef &&other) noexcept
    : opaque_(other.opaque_) {
  other.opaque_ = nullptr;
}

PipelineStateRef &
PipelineStateRef::operator=(PipelineStateRef &&other) noexcept {
  if (this != &other) {
    if (opaque_ != nullptr) {
      id<MTLComputePipelineState> pso =
          (__bridge_transfer id<MTLComputePipelineState>)opaque_;
      (void)pso;
    }
    opaque_ = other.opaque_;
    other.opaque_ = nullptr;
  }
  return *this;
}

// Helper that lets only this file construct a PipelineStateRef from a +1
// owned id<MTLComputePipelineState>. Keeps the ARC bookkeeping local.
class PipelineStateBuilder {
public:
  static std::shared_ptr<PipelineStateRef>
  Take(id<MTLComputePipelineState> pso) {
    auto ref = std::make_shared<PipelineStateRef>();
    // __bridge_retained: transfer ownership from ARC to opaque +1.
    ref->opaque_ = (__bridge_retained void *)pso;
    return ref;
  }
};

absl::StatusOr<CompiledPipeline>
CompileAndProbe(void *device_opaque, absl::string_view msl,
                absl::string_view entry_name, int64_t descriptor_thread_hint) {
  @autoreleasepool {
    if (device_opaque == nullptr) {
      return absl::InvalidArgumentError("CompileAndProbe: null device handle.");
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_opaque;
    if (descriptor_thread_hint < 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CompileAndProbe: descriptor_thread_hint must be >= 1; got ",
          descriptor_thread_hint));
    }

    NSString *source_ns = [[NSString alloc] initWithBytes:msl.data()
                                                   length:msl.size()
                                                 encoding:NSUTF8StringEncoding];
    if (source_ns == nil) {
      return absl::InvalidArgumentError(
          "CompileAndProbe: MSL source is not valid UTF-8.");
    }
    NSString *fn_ns = [[NSString alloc] initWithBytes:entry_name.data()
                                               length:entry_name.size()
                                             encoding:NSUTF8StringEncoding];
    if (fn_ns == nil) {
      return absl::InvalidArgumentError(
          "CompileAndProbe: entry name is not valid UTF-8.");
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = NO;
    NSError *err = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:source_ns options:opts error:&err];
    if (library == nil) {
      return absl::InternalError(
          DescribeError(err, "CompileAndProbe: newLibraryWithSource"));
    }

    id<MTLFunction> fn = [library newFunctionWithName:fn_ns];
    if (fn == nil) {
      return absl::NotFoundError(absl::StrCat("CompileAndProbe: entry point '",
                                              entry_name,
                                              "' not found in compiled MSL."));
    }

    MTLComputePipelineDescriptor *desc =
        [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    // Compiler hint: we won't dispatch wider than this, so the register
    // allocator can spend more registers per thread. The resulting PSO may
    // still report a lower ceiling — the caller compares against
    // pso.maxTotalThreadsPerThreadgroup and retries if it doesn't fit.
    desc.maxTotalThreadsPerThreadgroup =
        static_cast<NSUInteger>(descriptor_thread_hint);

    err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithDescriptor:desc
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&err];
    if (pso == nil) {
      return absl::InternalError(
          DescribeError(err, "CompileAndProbe: newComputePipelineState"));
    }

    CompiledPipeline result;
    result.pso = PipelineStateBuilder::Take(pso);
    result.pso_max_threads =
        static_cast<int64_t>([pso maxTotalThreadsPerThreadgroup]);
    return result;
  }
}

} // namespace metal
} // namespace stream_executor
