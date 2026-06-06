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

#include "xla/backends/metal/runtime/metal_triangular_solve_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::metal {

// Concrete holder for the cached MPS kernel — forward-declared in the
// header to keep `<MetalPerformanceShaders.h>` out of public headers.
struct MetalTriangularSolveKernel {
  MPSMatrixSolveTriangular* trsm;
};

namespace {

MPSDataType ToMpsDataType(PrimitiveType type) {
  switch (type) {
    case F32:
      return MPSDataTypeFloat32;
    // MPSMatrixSolveTriangular asserts "Only MPSDataTypeFloat32 is supported"
    // at encode time, so we refuse other dtypes (including C64) here rather
    // than crash the process.
    default:
      return MPSDataTypeInvalid;
  }
}

int64_t ElementSizeBytes(PrimitiveType type) {
  switch (type) {
    case F32:
      return 4;
    default:
      return 0;
  }
}

}  // namespace

MetalTriangularSolveThunk::MetalTriangularSolveThunk(
    ThunkInfo thunk_info, const TriangularSolveOptions& options,
    PrimitiveType element_type, BufferAllocation::Slice a_slice, Shape a_shape,
    BufferAllocation::Slice b_inout_slice, Shape b_shape, int64_t batch_size,
    int64_t m, int64_t num_rhs)
    : Thunk(Kind::kTriangularSolve, std::move(thunk_info)),
      options_(options),
      element_type_(element_type),
      a_slice_(a_slice),
      a_shape_(std::move(a_shape)),
      b_inout_slice_(b_inout_slice),
      b_shape_(std::move(b_shape)),
      batch_size_(batch_size),
      m_(m),
      num_rhs_(num_rhs) {}

MetalTriangularSolveThunk::~MetalTriangularSolveThunk() = default;

std::string MetalTriangularSolveThunk::ToString(int /*indent*/) const {
  return absl::StrFormat(
      "metal_triangular_solve(batch=%d, M=%d, num_rhs=%d, dtype=%d, "
      "left=%d, lower=%d, transpose=%d, unit=%d), profile_annotation=%s",
      static_cast<int>(batch_size_), static_cast<int>(m_),
      static_cast<int>(num_rhs_), static_cast<int>(element_type_),
      static_cast<int>(options_.left_side()),
      static_cast<int>(options_.lower()),
      static_cast<int>(options_.transpose_a()),
      static_cast<int>(options_.unit_diagonal()), thunk_info().profile_annotation);
}

gpu::Thunk::BufferUses MetalTriangularSolveThunk::buffer_uses() const {
  return {
      BufferUse::Read(a_slice_, a_shape_),
      BufferUse::Write(b_inout_slice_, b_shape_),
  };
}

absl::Status MetalTriangularSolveThunk::EnsureKernel(
    stream_executor::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  MPSDataType dtype = ToMpsDataType(element_type_);
  if (dtype == MPSDataTypeInvalid) {
    // Complex trsms should have been expanded into matmul+select sequences
    // by MetalExpandComplexTriangularSolve before reaching here; if one
    // slips through, fail loudly rather than crash inside MPS.
    return absl::UnimplementedError(absl::StrCat(
        "MetalTriangularSolveThunk: unsupported element type ",
        static_cast<int>(element_type_),
        " (MPSMatrixSolveTriangular asserts F32-only; complex trsms must "
        "be expanded upstream)"));
  }

  // Layout swap: XLA's Fortran-laid-out (column-major) operands are
  // byte-equivalent to MPS's row-major view of the transposed matrix, so
  // the BLAS layout-flip identity gives:
  //   right     = options.left_side    (XLA A·X=B becomes X^T·A^T=B^T)
  //   upper     = options.lower        (A's transpose flips uplo)
  //   transpose = (TRANSPOSE or ADJOINT)  — for real types, A^H == A^T,
  //               so ADJOINT folds in cleanly; complex ADJOINT can't reach
  //               here because MPS rejected complex above.
  const bool right = options_.left_side();
  const bool upper = options_.lower();
  const bool mps_transpose =
      options_.transpose_a() == TriangularSolveOptions::TRANSPOSE ||
      options_.transpose_a() == TriangularSolveOptions::ADJOINT;
  const bool unit = options_.unit_diagonal();

  auto* metal_executor =
      static_cast<stream_executor::metal::MetalExecutor*>(executor);
  id<MTLDevice> device = metal_executor->device();
  if (device == nil) {
    return absl::FailedPreconditionError(
        "MetalTriangularSolveThunk: executor has no MTLDevice.");
  }

  MPSMatrixSolveTriangular* trsm = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:device
                 right:right
                 upper:upper
             transpose:mps_transpose
                  unit:unit
                 order:static_cast<NSUInteger>(m_)
       numberOfRightHandSides:static_cast<NSUInteger>(num_rhs_)
                 alpha:1.0];
  if (trsm == nil) {
    return absl::InternalError(
        "MetalTriangularSolveThunk: MPSMatrixSolveTriangular init returned "
        "nil");
  }
  // Single-matrix encodes; batching is done by looping in
  // ExecuteOnStream with per-batch MPSMatrix offsets.
  trsm.batchSize = 1;

  kernel_ = std::make_unique<MetalTriangularSolveKernel>();
  kernel_->trsm = trsm;
  return absl::OkStatus();
}

absl::Status MetalTriangularSolveThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  stream_executor::Stream* stream = params.stream;
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "MetalTriangularSolveThunk::ExecuteOnStream: null stream");
  }
  auto* metal_stream =
      dynamic_cast<stream_executor::metal::MetalStream*>(stream);
  if (metal_stream == nullptr) {
    return absl::InvalidArgumentError(
        "MetalTriangularSolveThunk::ExecuteOnStream: stream is not a "
        "MetalStream");
  }
  auto* metal_executor =
      static_cast<stream_executor::metal::MetalExecutor*>(stream->parent());

  // Resolve operand addresses.
  if (params.buffer_allocations == nullptr) {
    return absl::InvalidArgumentError(
        "MetalTriangularSolveThunk::ExecuteOnStream: null buffer_allocations");
  }
  stream_executor::DeviceAddressBase a_addr =
      params.buffer_allocations->GetDeviceAddress(a_slice_);
  stream_executor::DeviceAddressBase b_addr =
      params.buffer_allocations->GetDeviceAddress(b_inout_slice_);

  // The MPS kernel object is built once at first execution and re-used. It
  // holds internal scratch; Apple's docs say concurrent encodes of the same
  // kernel are unsafe even to distinct command buffers, so serialise via
  // encode_mu_ across multi-stream re-executions of this thunk.
  absl::MutexLock lock(&encode_mu_);
  TF_RETURN_IF_ERROR(EnsureKernel(stream->parent()));

  auto* allocator = metal_executor->allocator();
  auto resolved_a = allocator->Resolve(a_addr.opaque());
  if (!resolved_a.has_value()) {
    return absl::InvalidArgumentError(
        "MetalTriangularSolveThunk: A pointer not owned by the executor's "
        "allocator");
  }
  auto resolved_b = allocator->Resolve(b_addr.opaque());
  if (!resolved_b.has_value()) {
    return absl::InvalidArgumentError(
        "MetalTriangularSolveThunk: B pointer not owned by the executor's "
        "allocator");
  }

  const MPSDataType dtype = ToMpsDataType(element_type_);
  const int64_t elem_size = ElementSizeBytes(element_type_);

  // MPS descriptor (rows, columns) is the row-major view of the
  // Fortran-laid-out tensor. rowBytes = MPS columns * elem_size.
  // right=YES → B/X is (num_rhs, order); right=NO → B/X is (order, num_rhs).
  const bool right_side = options_.left_side();
  NSUInteger b_rows = right_side ? num_rhs_ : m_;
  NSUInteger b_cols = right_side ? m_ : num_rhs_;
  NSUInteger b_row_bytes = b_cols * elem_size;
  NSUInteger b_matrix_bytes = b_rows * b_row_bytes;
  NSUInteger a_row_bytes = m_ * elem_size;
  NSUInteger a_matrix_bytes = m_ * a_row_bytes;

  // Single-matrix descriptors. Batching is done by re-wrapping the buffer
  // at successive offsets and encoding the kernel once per batch — MPS's
  // built-in batched dispatch via descriptor.matrices + kernel.batchSize
  // didn't process tightly-packed (rowBytes = cols * elem_size) Fortran-
  // layout batches correctly.
  MPSMatrixDescriptor* a_desc =
      [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(m_)
                                            columns:static_cast<NSUInteger>(m_)
                                           rowBytes:a_row_bytes
                                           dataType:dtype];
  MPSMatrixDescriptor* b_desc = [MPSMatrixDescriptor
      matrixDescriptorWithRows:b_rows
                       columns:b_cols
                      rowBytes:b_row_bytes
                      dataType:dtype];

  MPSMatrixSolveTriangular* trsm = kernel_->trsm;
  auto encoded = metal_stream->EncodeWithCommandBuffer(
      /*label=*/"MetalTRSM",
      /*op_name=*/"MetalTriangularSolveThunk",
      [&](id<MTLCommandBuffer> cmd_buf)
          -> absl::StatusOr<id<MTLCommandBuffer>> {
        for (NSUInteger i = 0; i < static_cast<NSUInteger>(batch_size_); ++i) {
          MPSMatrix* a_one = [[MPSMatrix alloc]
              initWithBuffer:resolved_a->buffer
                      offset:resolved_a->offset + i * a_matrix_bytes
                  descriptor:a_desc];
          MPSMatrix* b_one = [[MPSMatrix alloc]
              initWithBuffer:resolved_b->buffer
                      offset:resolved_b->offset + i * b_matrix_bytes
                  descriptor:b_desc];
          if (a_one == nil || b_one == nil) {
            return absl::InternalError(
                "MetalTriangularSolveThunk: MPSMatrix init returned nil");
          }
          [trsm encodeToCommandBuffer:cmd_buf
                         sourceMatrix:a_one
                  rightHandSideMatrix:b_one
                       solutionMatrix:b_one];
        }
        return cmd_buf;
      });
  if (!encoded.ok()) return encoded.status();
  return absl::OkStatus();
}

}  // namespace xla::metal
