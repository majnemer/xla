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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_FFT_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_FFT_H_

#include <complex>
#include <cstdint>
#include <memory>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::metal {

class MetalExecutor;

// MPSGraph-backed fft::FftSupport for Metal. Plans compile an MPSGraph once
// (forward or inverse FFT op, plus real-to-Hermitean / Hermitean-to-real for
// the R2C / C2R variants) and re-encode it cheaply per DoFft call onto the
// stream's command buffer. The plan reports
// PostFftScaleFactor(inverse_type, N) == 1 so FftThunk skips the post-FFT
// BLAS normalisation (Metal has no BLAS plugin); MPSGraph's
// MPSGraphFFTScalingMode.Size produces the already-normalised result.
//
// F64 (kZ2Z*, kD2Z, kZ2D) is unsupported — Apple silicon has no F64 in MPS;
// CreateBatchedPlanWithScratchAllocator returns nullptr in those cases.
class MetalFft final : public fft::FftSupport {
 public:
  explicit MetalFft(MetalExecutor* executor) : executor_(executor) {}
  ~MetalFft() override = default;

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

  MetalExecutor* executor() const { return executor_; }

 private:
  MetalExecutor* executor_;

  MetalFft(const MetalFft&) = delete;
  MetalFft& operator=(const MetalFft&) = delete;
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_FFT_H_
