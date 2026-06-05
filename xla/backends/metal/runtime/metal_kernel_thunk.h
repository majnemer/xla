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

#ifndef XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_THUNK_H_
#define XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_THUNK_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/metal/runtime/metal_kernel_artifact.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/stream_executor.h"

// Forward declaration so this header stays free of <Metal/Metal.h>.
namespace stream_executor::metal {
class MetalKernel;
}  // namespace stream_executor::metal

namespace xla::metal {

// Thunk that dispatches a Metal compute kernel using a precompiled
// MTLComputePipelineState. The PSO is built once at compile time by
// MetalCompiler's per-fusion retry loop (see CompileAndProbe) and stored
// inside `artifact`. At runtime, ExecuteOnStream resolves the argument slices
// to device pointers and launches via the existing MetalStream::LaunchKernel
// path — no MTLLibrary lookup, no MSL compilation.
class MetalKernelThunk : public gpu::Thunk {
 public:
  // Constructs the thunk with the static argument slices. The compile-time
  // PSO artifact is installed by SetArtifact() after the per-fusion retry
  // loop in MetalCompiler::CompileToBackendResult builds the PSO.
  MetalKernelThunk(ThunkInfo thunk_info,
                   const emitters::KernelArguments& kernel_arguments);
  ~MetalKernelThunk() override;

  MetalKernelThunk(const MetalKernelThunk&) = delete;
  MetalKernelThunk& operator=(const MetalKernelThunk&) = delete;

  // Installs the precompiled PSO + launch metadata. Called once, before the
  // executable is handed to the runtime. ExecuteOnStream fails if not called.
  void SetArtifact(std::unique_ptr<MetalKernelArtifact> artifact);

  std::string ToString(int indent) const override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  gpu::Thunk::BufferUses buffer_uses() const override;

 private:
  std::unique_ptr<MetalKernelArtifact> artifact_;
  std::vector<ShapedSlice> args_;
  std::vector<bool> written_;

  // Per-executor MetalKernel cache. The MetalKernel wraps the artifact's PSO
  // plus the executor-specific metadata so it can be passed to
  // MetalStream::LaunchKernel via the existing void*-function interface.
  // Mirror of gpu::KernelThunk::kernel_cache_.
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<stream_executor::metal::MetalKernel>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_RUNTIME_METAL_KERNEL_THUNK_H_
