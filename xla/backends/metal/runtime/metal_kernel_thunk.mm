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

#include "xla/backends/metal/runtime/metal_kernel_thunk.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/metal/runtime/metal_kernel_artifact.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/metal/metal_pso_probe.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::metal {

MetalKernelThunk::MetalKernelThunk(
    ThunkInfo thunk_info,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()) {}

MetalKernelThunk::~MetalKernelThunk() = default;

void MetalKernelThunk::SetArtifact(
    std::unique_ptr<MetalKernelArtifact> artifact) {
  artifact_ = std::move(artifact);
}

std::string MetalKernelThunk::ToString(int /*indent*/) const {
  if (artifact_ == nullptr) {
    return absl::StrFormat("metal_kernel=<unresolved>, profile_annotation=%s",
                           thunk_info().profile_annotation);
  }
  const auto& dims = artifact_->launch_dimensions();
  return absl::StrFormat(
      "metal_kernel=%s, profile_annotation=%s, threads=(%d,%d,%d), "
      "blocks=(%d,%d,%d)",
      artifact_->entry_name(), thunk_info().profile_annotation,
      dims.thread_counts_per_block().x, dims.thread_counts_per_block().y,
      dims.thread_counts_per_block().z, dims.block_counts().x,
      dims.block_counts().y, dims.block_counts().z);
}

gpu::Thunk::BufferUses MetalKernelThunk::buffer_uses() const {
  gpu::Thunk::BufferUses uses;
  uses.reserve(args_.size());
  for (size_t i = 0; i < args_.size(); ++i) {
    if (written_[i]) {
      uses.push_back(BufferUse::Write(args_[i].slice, args_[i].shape));
    } else {
      uses.push_back(BufferUse::Read(args_[i].slice, args_[i].shape));
    }
  }
  return uses;
}

absl::Status MetalKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (artifact_ == nullptr) {
    return absl::FailedPreconditionError(
        "MetalKernelThunk::ExecuteOnStream: artifact not installed; "
        "SetArtifact() was not called before execution.");
  }
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  auto* metal_executor =
      dynamic_cast<stream_executor::metal::MetalExecutor*>(executor);
  if (metal_executor == nullptr) {
    return absl::FailedPreconditionError(
        "MetalKernelThunk::ExecuteOnStream: stream's executor is not a "
        "MetalExecutor.");
  }

  // Get or build the per-executor MetalKernel wrapping the artifact's PSO.
  stream_executor::metal::MetalKernel* kernel = nullptr;
  {
    absl::MutexLock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    if (it == kernel_cache_.end()) {
      if (artifact_->pipeline_state() == nullptr ||
          artifact_->pipeline_state()->empty()) {
        return absl::InternalError(absl::StrFormat(
            "MetalKernelThunk::ExecuteOnStream: artifact for '%s' has no "
            "compiled pipeline state.",
            artifact_->entry_name()));
      }
      id<MTLComputePipelineState> pso =
          (__bridge id<MTLComputePipelineState>)
              artifact_->pipeline_state()->opaque();
      TF_ASSIGN_OR_RETURN(auto new_kernel,
                          stream_executor::metal::MetalKernel::CreateFromPSO(
                              metal_executor, pso, artifact_->arity()));
      new_kernel->set_name(artifact_->entry_name());
      kernel = new_kernel.get();
      kernel_cache_.emplace(executor, std::move(new_kernel));
    } else {
      kernel = it->second.get();
    }
  }

  // Resolve slices to device addresses and call the standard kernel launch
  // path. ExecuteKernelOnStream packs args and dispatches via
  // se::Kernel::Launch -> MetalKernel::Launch -> MetalStream::LaunchKernel.
  absl::InlinedVector<se::KernelArg, 4> kernel_args;
  kernel_args.reserve(args_.size());
  for (size_t i = 0; i < args_.size(); ++i) {
    kernel_args.push_back(
        params.buffer_allocations->GetDeviceAddress(args_[i].slice));
  }
  return gpu::ExecuteKernelOnStream(
      *kernel,
      absl::Span<se::KernelArg>(kernel_args.data(), kernel_args.size()),
      artifact_->launch_dimensions(),
      /*cluster_dim=*/std::nullopt, stream);
}

}  // namespace xla::metal
