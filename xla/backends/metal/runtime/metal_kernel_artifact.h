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

#ifndef XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_ARTIFACT_H_
#define XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_ARTIFACT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/metal/metal_pso_probe.h"

namespace xla::metal {

// Precompiled artifact for one fusion: the MTLComputePipelineState (built at
// compile time by the MetalCompiler's per-fusion retry loop), the entry-point
// name, the launch dimensions chosen at the same time as the PSO, and the
// argument arity. Owned directly by MetalKernelThunk — no separate registry.
//
// The MTLComputePipelineState is held through PipelineStateRef so this header
// stays free of <Metal/Metal.h>; the thunk bridges back to id<>
// inside its .mm implementation.
class MetalKernelArtifact {
 public:
  MetalKernelArtifact(std::string entry_name, unsigned arity,
                      gpu::LaunchDimensions launch_dims,
                      std::shared_ptr<stream_executor::metal::PipelineStateRef>
                          pipeline_state)
      : entry_name_(std::move(entry_name)),
        arity_(arity),
        launch_dims_(launch_dims),
        pipeline_state_(std::move(pipeline_state)) {}

  const std::string& entry_name() const { return entry_name_; }
  unsigned arity() const { return arity_; }
  const gpu::LaunchDimensions& launch_dimensions() const {
    return launch_dims_;
  }
  const stream_executor::metal::PipelineStateRef* pipeline_state() const {
    return pipeline_state_.get();
  }

 private:
  std::string entry_name_;
  unsigned arity_;
  gpu::LaunchDimensions launch_dims_;
  std::shared_ptr<stream_executor::metal::PipelineStateRef> pipeline_state_;
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_ARTIFACT_H_
