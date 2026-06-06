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

#ifndef XLA_BACKENDS_METAL_RUNTIME_METAL_TRIANGULAR_SOLVE_THUNK_H_
#define XLA_BACKENDS_METAL_RUNTIME_METAL_TRIANGULAR_SOLVE_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::metal {

// Opaque holder for the cached MPSMatrixSolveTriangular kernel object.
// Defined in the .mm file to keep this header free of MPS headers.
struct MetalTriangularSolveKernel;

// Metal-specific thunk for kTriangularSolve. Wraps a single
// MPSMatrixSolveTriangular kernel object built at thunk construction with
// every option (right/upper/transpose/unit/order/num_rhs/batch_size) baked
// in, then encodes that kernel onto the stream's command buffer at run time
// under a per-thunk mutex.
//
// XLA's transpose_a == ADJOINT must be lowered upstream — this thunk only
// understands NO_TRANSPOSE / TRANSPOSE. Metal's pipeline runs the
// MetalLowerAdjointTriangularSolve pass for that.
class MetalTriangularSolveThunk : public gpu::Thunk {
 public:
  MetalTriangularSolveThunk(ThunkInfo thunk_info,
                            const TriangularSolveOptions& options,
                            PrimitiveType element_type,
                            BufferAllocation::Slice a_slice, Shape a_shape,
                            BufferAllocation::Slice b_inout_slice,
                            Shape b_shape, int64_t batch_size, int64_t m,
                            int64_t num_rhs);
  ~MetalTriangularSolveThunk() override;

  MetalTriangularSolveThunk(const MetalTriangularSolveThunk&) = delete;
  MetalTriangularSolveThunk& operator=(const MetalTriangularSolveThunk&) =
      delete;

  std::string ToString(int indent) const override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  gpu::Thunk::BufferUses buffer_uses() const override;

 private:
  // Lazily builds the cached kernel on first execute. Returns OK on success;
  // returns an error if the executor's MTLDevice is unavailable or MPS
  // refuses to construct the kernel. Idempotent.
  absl::Status EnsureKernel(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(encode_mu_);

  const TriangularSolveOptions options_;
  const PrimitiveType element_type_;
  const BufferAllocation::Slice a_slice_;
  const Shape a_shape_;
  const BufferAllocation::Slice b_inout_slice_;
  const Shape b_shape_;
  const int64_t batch_size_;
  const int64_t m_;          // A is M x M.
  const int64_t num_rhs_;    // The non-M dim of B.

  // Serialises concurrent ExecuteOnStream calls. The kernel object holds
  // internal scratch memory; per Apple's docs, concurrent encodes of the
  // same kernel are unsafe even to distinct command buffers.
  absl::Mutex encode_mu_;
  std::unique_ptr<MetalTriangularSolveKernel> kernel_ ABSL_GUARDED_BY(encode_mu_);
};

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_RUNTIME_METAL_TRIANGULAR_SOLVE_THUNK_H_
