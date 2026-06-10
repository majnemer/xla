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

#ifndef XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_THUNK_H_
#define XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_THUNK_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/metal/runtime/metal_graph_artifact.h"
#include "xla/service/shaped_slice.h"

namespace xla::metal {

// Executes a compiled MPSGraph region (__metal_graph fusion). Feeds bind in
// fusion operand order; the single result binds to the fusion's slice. Every
// address it touches is slice-derived, so buffer_uses() fully describes its
// footprint for the dependency DAG and command-buffer introspection.
//
// The artifact (compiled at MetalCompiler compile time, like MSL PSOs) is
// installed via SetArtifact before the executable is handed to the runtime.
class MetalGraphThunk : public gpu::Thunk {
 public:
  MetalGraphThunk(ThunkInfo thunk_info, std::vector<ShapedSlice> feeds,
                  ShapedSlice result);
  ~MetalGraphThunk() override;

  MetalGraphThunk(const MetalGraphThunk&) = delete;
  MetalGraphThunk& operator=(const MetalGraphThunk&) = delete;

  void SetArtifact(std::unique_ptr<MetalGraphArtifact> artifact);

  std::string ToString(int indent) const override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  gpu::Thunk::BufferUses buffer_uses() const override;

  absl::StatusOr<gpu::ThunkProto> ToProto() const override {
    return absl::UnimplementedError(
        "MetalGraphThunk::ToProto is not implemented.");
  }

 private:
  std::unique_ptr<MetalGraphArtifact> artifact_;
  std::vector<ShapedSlice> feeds_;
  ShapedSlice result_;

  // MPSGraphExecutable's encode thread-safety is undocumented; serialize
  // concurrent encodes of this executable. Mirrors
  // MetalTriangularSolveThunk::encode_mu_.
  absl::Mutex encode_mu_;
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_RUNTIME_METAL_GRAPH_THUNK_H_
