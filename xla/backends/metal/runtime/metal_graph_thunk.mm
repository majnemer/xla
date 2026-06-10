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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "xla/backends/metal/runtime/metal_graph_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/metal/runtime/metal_graph_artifact.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::metal {
namespace {

NSArray<NSNumber*>* ToNSShape(absl::Span<const int64_t> dims) {
  NSMutableArray<NSNumber*>* shape =
      [NSMutableArray arrayWithCapacity:dims.size()];
  for (int64_t d : dims) {
    [shape addObject:@(d)];
  }
  return shape;
}

// Wraps a slice-resolved device pointer as MPSGraphTensorData. Routed
// through MPSNDArray because BFC sub-slices carry offsets that the direct
// MTLBuffer initializer cannot express (same as MetalFft's MakeTensorData).
absl::StatusOr<MPSGraphTensorData*> MakeTensorData(
    stream_executor::metal::MetalAllocator* allocator, const void* ptr,
    const MetalGraphTensorDescriptor& desc) {
  auto resolved = allocator->Resolve(ptr);
  if (!resolved.has_value()) {
    return absl::InvalidArgumentError(
        "MetalGraphThunk: buffer not owned by the executor's allocator");
  }
  MPSNDArrayDescriptor* nd_desc = [MPSNDArrayDescriptor
      descriptorWithDataType:(MPSDataType)desc.mps_data_type
                       shape:ToNSShape(desc.physical_dims)];
  if (nd_desc == nil) {
    return absl::InternalError(
        "MetalGraphThunk: MPSNDArrayDescriptor init returned nil");
  }
  // XLA buffers are densely packed; without this MPSNDArray pads rowBytes to
  // 16 and rejects buffers smaller than the padded size.
  nd_desc.preferPackedRows = YES;
  MPSNDArray* ndarray = [[MPSNDArray alloc] initWithBuffer:resolved->buffer
                                                    offset:resolved->offset
                                                descriptor:nd_desc];
  if (ndarray == nil) {
    return absl::InternalError("MetalGraphThunk: MPSNDArray init returned nil");
  }
  MPSGraphTensorData* data =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray];
  if (data == nil) {
    return absl::InternalError(
        "MetalGraphThunk: MPSGraphTensorData init returned nil");
  }
  return data;
}

}  // namespace

MetalGraphThunk::MetalGraphThunk(ThunkInfo thunk_info,
                                 std::vector<ShapedSlice> feeds,
                                 ShapedSlice result)
    : Thunk(Kind::kKernel, std::move(thunk_info)),
      feeds_(std::move(feeds)),
      result_(std::move(result)) {}

MetalGraphThunk::~MetalGraphThunk() = default;

void MetalGraphThunk::SetArtifact(std::unique_ptr<MetalGraphArtifact> artifact) {
  artifact_ = std::move(artifact);
}

std::string MetalGraphThunk::ToString(int /*indent*/) const {
  return absl::StrFormat(
      "metal_graph=%s, feeds=%d, profile_annotation=%s",
      artifact_ == nullptr ? "<unresolved>" : "<compiled>", feeds_.size(),
      thunk_info().profile_annotation);
}

gpu::Thunk::BufferUses MetalGraphThunk::buffer_uses() const {
  gpu::Thunk::BufferUses uses;
  uses.reserve(feeds_.size() + 1);
  for (const ShapedSlice& feed : feeds_) {
    uses.push_back(BufferUse::Read(feed.slice, feed.shape));
  }
  uses.push_back(BufferUse::Write(result_.slice, result_.shape));
  return uses;
}

absl::Status MetalGraphThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (artifact_ == nullptr || artifact_->executable.empty()) {
    return absl::FailedPreconditionError(
        "MetalGraphThunk::ExecuteOnStream: artifact not installed; "
        "SetArtifact() was not called before execution.");
  }
  if (artifact_->feeds.size() != feeds_.size()) {
    return absl::InternalError(
        "MetalGraphThunk::ExecuteOnStream: feed descriptor count mismatch.");
  }
  se::Stream* stream = params.stream;
  auto* metal_executor =
      dynamic_cast<stream_executor::metal::MetalExecutor*>(stream->parent());
  if (metal_executor == nullptr) {
    return absl::FailedPreconditionError(
        "MetalGraphThunk::ExecuteOnStream: stream's executor is not a "
        "MetalExecutor.");
  }
  stream_executor::metal::MetalAllocator* allocator =
      metal_executor->allocator();
  auto* metal_stream = static_cast<stream_executor::metal::MetalStream*>(stream);

  @autoreleasepool {
    NSMutableArray<MPSGraphTensorData*>* inputs =
        [NSMutableArray arrayWithCapacity:feeds_.size()];
    for (size_t i = 0; i < feeds_.size(); ++i) {
      const void* ptr =
          params.buffer_allocations->GetDeviceAddress(feeds_[i].slice).opaque();
      TF_ASSIGN_OR_RETURN(
          MPSGraphTensorData * data,
          MakeTensorData(allocator, ptr, artifact_->feeds[i]));
      [inputs addObject:data];
    }
    void* result_ptr =
        params.buffer_allocations->GetDeviceAddress(result_.slice).opaque();
    TF_ASSIGN_OR_RETURN(
        MPSGraphTensorData * result_data,
        MakeTensorData(allocator, result_ptr, artifact_->result));
    NSArray<MPSGraphTensorData*>* outputs = @[ result_data ];

    MPSGraphExecutable* executable =
        (__bridge MPSGraphExecutable*)artifact_->executable.opaque();

    absl::MutexLock lock(encode_mu_);
    TF_ASSIGN_OR_RETURN(
        id<MTLCommandBuffer> encoded,
        metal_stream->EncodeWithCommandBuffer(
            /*label=*/"MetalGraph",
            /*op_name=*/"MetalGraphThunk::ExecuteOnStream",
            [&](id<MTLCommandBuffer> cmd_buf)
                -> absl::StatusOr<id<MTLCommandBuffer>> {
              MPSCommandBuffer* mps_cmd_buf =
                  [MPSCommandBuffer commandBufferWithCommandBuffer:cmd_buf];
              MPSGraphExecutableExecutionDescriptor* exec_desc =
                  [[MPSGraphExecutableExecutionDescriptor alloc] init];
              [executable encodeToCommandBuffer:mps_cmd_buf
                                    inputsArray:inputs
                                   resultsArray:outputs
                            executionDescriptor:exec_desc];
              // MPSGraph may commitAndContinue internally; track the final
              // underlying buffer as the stream's new tail.
              return [mps_cmd_buf commandBuffer];
            }));
    (void)encoded;
  }
  return absl::OkStatus();
}

}  // namespace xla::metal
