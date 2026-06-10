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

#ifndef XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_ARTIFACT_H_
#define XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_ARTIFACT_H_

#include <cstdint>
#include <vector>

#include "xla/stream_executor/metal/metal_graph_executable.h"

namespace xla::metal {

// How one feed/result buffer binds to an MPSGraph tensor: dimension sizes in
// physical (slowest-to-fastest) order — the dense interpretation of the XLA
// buffer — plus the MPSDataType raw value. Pure C++ so compiler-side code
// can carry it; the thunk's .mm builds MPSNDArray descriptors from it.
struct MetalGraphTensorDescriptor {
  std::vector<int64_t> physical_dims;
  uint32_t mps_data_type = 0;
};

// Compiled MPSGraph region: the executable plus binding descriptors, feeds
// in fusion operand order. Built by BuildMetalGraphArtifact at compile time;
// owned by MetalGraphThunk.
struct MetalGraphArtifact {
  stream_executor::metal::GraphExecutableRef executable;
  std::vector<MetalGraphTensorDescriptor> feeds;
  MetalGraphTensorDescriptor result;
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_ARTIFACT_H_
