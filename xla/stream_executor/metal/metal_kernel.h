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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

// Kernel implementation wrapping an MTLComputePipelineState.
//
// Create() looks up the named entry-point MTLFunction in the supplied
// library and constructs a compute pipeline state. The library is owned and
// JIT-cached by MetalExecutor (mirror of CudaExecutor::gpu_binary_to_module_):
// MSL compilation runs once per source blob; per-kernel work here is symbol
// resolution + PSO codegen.
//
// v1 status: pipeline-state construction works; Launch returns Unimplemented
// until MetalStream::LaunchKernel encoding lands.
class MetalKernel : public Kernel {
 public:
  static absl::StatusOr<std::unique_ptr<MetalKernel>> Create(
      MetalExecutor* executor, id<MTLLibrary> library,
      absl::string_view entry_point, unsigned arity);

  // Constructs a MetalKernel from an externally-built (and CFRetained)
  // MTLComputePipelineState. Used by MetalKernelThunk to wrap a precompiled
  // PSO produced by xla::metal::CompileAndProbe at compile time, bypassing
  // both MSL compilation and PSO creation at runtime.
  //
  // `pso` must already be retained; this factory takes a +1 reference. The
  // resulting MetalKernel holds no MTLFunction (function() returns nil).
  static absl::StatusOr<std::unique_ptr<MetalKernel>> CreateFromPSO(
      MetalExecutor* executor, id<MTLComputePipelineState> pso, unsigned arity);

  ~MetalKernel() override;

  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  absl::Status Launch(const ThreadDim& thread_dims,
                      const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  // Underlying objects. Available only inside .mm consumers.
  id<MTLComputePipelineState> pipeline_state() const {
    return pipeline_state_;
  }
  id<MTLFunction> function() const { return function_; }

 private:
  MetalKernel(MetalExecutor* executor, id<MTLComputePipelineState> pso,
              id<MTLFunction> function, unsigned arity);

  MetalExecutor* executor_;
  id<MTLComputePipelineState> pipeline_state_;
  // Function is retained so reflection / re-pipelining stays possible if we
  // ever want it; it's also useful in diagnostics.
  id<MTLFunction> function_;
  unsigned arity_;

  MetalKernel(const MetalKernel&) = delete;
  MetalKernel& operator=(const MetalKernel&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_
