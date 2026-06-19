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

#include "xla/stream_executor/metal/metal_kernel.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace metal {
namespace {

// Returns the localized description of an NSError as a UTF-8 std::string,
// or a fallback if the error pointer is nil.
std::string ErrorMessage(NSError *error) {
  if (error == nil || [error localizedDescription] == nil) {
    return "(no error description)";
  }
  return [[error localizedDescription] UTF8String];
}

absl::StatusOr<NSString *> MakeNSString(absl::string_view sv,
                                        absl::string_view what) {
  NSString *s = [[NSString alloc] initWithBytes:sv.data()
                                         length:sv.size()
                                       encoding:NSUTF8StringEncoding];
  if (s == nil) {
    return absl::InvalidArgumentError(
        absl::StrCat("MetalKernel::Create: ", what, " is not valid UTF-8."));
  }
  return s;
}

} // namespace

MetalKernel::MetalKernel(MetalExecutor *executor,
                         id<MTLComputePipelineState> pso,
                         id<MTLFunction> function, unsigned arity)
    : executor_(executor), pipeline_state_(pso), function_(function),
      arity_(arity) {}

MetalKernel::~MetalKernel() {
  if (executor_ != nullptr) {
    executor_->UnloadKernel(this);
  }
}

absl::StatusOr<std::unique_ptr<MetalKernel>>
MetalKernel::Create(MetalExecutor *executor, id<MTLLibrary> library,
                    absl::string_view entry_point, unsigned arity) {
  @autoreleasepool {
    id<MTLDevice> device = executor->device();
    if (device == nil) {
      return absl::FailedPreconditionError(
          "MetalKernel::Create: executor has no MTLDevice; was Init() called?");
    }
    if (library == nil) {
      return absl::InvalidArgumentError("MetalKernel::Create: library is nil.");
    }
    // PSO compilation autoreleases internal temporaries; bound them here.
    // function / pso are +1 owned (new*), so they survive the pool drain.
    auto entry_ns = MakeNSString(entry_point, "entry point");
    if (!entry_ns.ok()) {
      return entry_ns.status();
    }
    id<MTLFunction> function = [library newFunctionWithName:*entry_ns];
    if (function == nil) {
      return absl::NotFoundError(
          absl::StrCat("MetalKernel::Create: entry point '", entry_point,
                       "' not found in library."));
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pso == nil) {
      return absl::InternalError(absl::StrCat(
          "MetalKernel::Create: newComputePipelineStateWithFunction failed: ",
          ErrorMessage(error)));
    }
    // Codegen bakes DeviceDescription's threads_per_warp into launch sizing
    // and shuffle widths; fail loud if the PSO disagrees.
    const auto pso_simd = static_cast<int64_t>([pso threadExecutionWidth]);
    const int64_t expected_simd =
        executor->GetDeviceDescription().threads_per_warp();
    if (pso_simd != expected_simd) {
      return absl::FailedPreconditionError(absl::StrCat(
          "MetalKernel::Create: PSO threadExecutionWidth (", pso_simd,
          ") differs from DeviceDescription threads_per_warp (", expected_simd,
          ") for entry point '", entry_point, "'."));
    }
    return std::unique_ptr<MetalKernel>(
        new MetalKernel(executor, pso, function, arity));
  }
}

absl::StatusOr<std::unique_ptr<MetalKernel>>
MetalKernel::CreateFromPSO(MetalExecutor *executor,
                           id<MTLComputePipelineState> pso, unsigned arity) {
  @autoreleasepool {
    if (executor == nullptr) {
      return absl::InvalidArgumentError(
          "MetalKernel::CreateFromPSO: executor is null.");
    }
    if (pso == nil) {
      return absl::InvalidArgumentError(
          "MetalKernel::CreateFromPSO: pso is nil.");
    }
    const auto pso_simd = static_cast<int64_t>([pso threadExecutionWidth]);
    const int64_t expected_simd =
        executor->GetDeviceDescription().threads_per_warp();
    if (pso_simd != expected_simd) {
      return absl::FailedPreconditionError(absl::StrCat(
          "MetalKernel::CreateFromPSO: PSO threadExecutionWidth (", pso_simd,
          ") differs from DeviceDescription threads_per_warp (", expected_simd,
          ")."));
    }
    return std::unique_ptr<MetalKernel>(
        new MetalKernel(executor, pso, /*function=*/nil, arity));
  }
}

absl::StatusOr<int32_t> MetalKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim /*threads*/, size_t /*dynamic_shared_memory_bytes*/) const {
  // Metal has no per-SM occupancy query; return 1 to keep callers
  // conservative (kernel treated as fully occupying a core per launch).
  return 1;
}

absl::Status MetalKernel::Launch(const ThreadDim &thread_dims,
                                 const BlockDim &block_dims,
                                 const std::optional<ClusterDim> &cluster_dims,
                                 Stream *stream, const KernelArgs &args) {
  // Pack args unless already flat. Mirrors CudaKernel::Launch.
  auto launch = [this, stream, &cluster_dims, &thread_dims, &block_dims](
                    const KernelArgsPackedArrayBase &packed) -> absl::Status {
    const size_t expected =
        Arity() + (packed.number_of_shared_bytes() > 0 ? 1 : 0);
    if (packed.number_of_arguments() != expected) {
      return absl::InvalidArgumentError(absl::StrCat(
          "MetalKernel::Launch: kernel ", name(), " has ",
          packed.number_of_arguments(), " packed argument(s), expected ",
          expected, " (arity=", Arity(),
          ", shmem_bytes=", packed.number_of_shared_bytes(), ")."));
    }
    if (cluster_dims.has_value()) {
      return absl::UnimplementedError(
          "MetalKernel::Launch: cluster dimensions are not supported on "
          "Metal.");
    }
    auto *metal_stream = dynamic_cast<MetalStream *>(stream);
    if (metal_stream == nullptr) {
      return absl::InvalidArgumentError(
          "MetalKernel::Launch: stream is not a MetalStream.");
    }
    // Sized binding lets registry kernels pass by-value scalars (setBytes);
    // packers without size metadata keep the all-buffers contract.
    return metal_stream->LaunchKernelPacked(
        thread_dims, block_dims, this, name(), packed.argument_addresses(),
        packed.argument_sizes(),
        static_cast<int64_t>(packed.number_of_shared_bytes()));
  };

  if (auto *packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return launch(*packed);
  }
  if (auto *device_mem = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    const auto &pack = args_packing();
    if (!pack) {
      return absl::InternalError(
          "MetalKernel::Launch: kernel is missing a custom args packing for "
          "device-memory arguments.");
    }
    TF_ASSIGN_OR_RETURN(auto packed, pack(*this, *device_mem));
    return launch(*packed);
  }
  return absl::InternalError(
      "MetalKernel::Launch: unsupported KernelArgs type.");
}

} // namespace metal
} // namespace stream_executor
