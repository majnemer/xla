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

#include "xla/stream_executor/metal/metal_fft.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/metal/metal_allocator.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::metal {
namespace {

// Maps an XLA fft::Type onto MPSGraph descriptor settings + tensor dtypes.
// Returns false if the type is unsupported (F64 variants on Apple silicon).
struct FftCase {
  bool inverse;
  bool real_input;
  bool real_output;
  // True when the transform crosses the real/complex boundary in either
  // direction. R2C uses realToHermiteanFFT; C2R uses HermiteanToRealFFT.
  bool real_to_hermitean;
  bool hermitean_to_real;
};

std::optional<FftCase> Classify(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
      return FftCase{/*inverse=*/false, false, false, false, false};
    case fft::Type::kC2CInverse:
      return FftCase{/*inverse=*/true, false, false, false, false};
    case fft::Type::kR2C:
      return FftCase{/*inverse=*/false, true, false,
                     /*real_to_hermitean=*/true, false};
    case fft::Type::kC2R:
      return FftCase{/*inverse=*/true, false, true, false,
                     /*hermitean_to_real=*/true};
    // F64 variants are unsupported on Apple silicon.
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse:
    case fft::Type::kD2Z:
    case fft::Type::kZ2D:
      return std::nullopt;
    case fft::Type::kInvalid:
      return std::nullopt;
  }
}

// Builds the [batch, fft_dim_0, ..., fft_dim_R-1] shape array for a tensor.
// Matches the layout FftThunk hands us: dense, batch-major, unit-strided.
NSArray<NSNumber*>* MakeShape(int batch, const uint64_t* embed, int rank) {
  NSMutableArray<NSNumber*>* shape =
      [NSMutableArray arrayWithCapacity:rank + 1];
  [shape addObject:@(batch)];
  for (int i = 0; i < rank; ++i) {
    [shape addObject:@(static_cast<NSInteger>(embed[i]))];
  }
  return shape;
}

NSArray<NSNumber*>* MakeAxes(int rank) {
  NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:rank];
  for (int i = 0; i < rank; ++i) {
    [axes addObject:@(i + 1)];  // skip leading batch axis.
  }
  return axes;
}

uint64_t Prod(const uint64_t* xs, int n) {
  uint64_t p = 1;
  for (int i = 0; i < n; ++i) p *= xs[i];
  return p;
}

// MetalFftPlan: holds the compiled MPSGraphExecutable plus the metadata
// needed to construct MPSGraphTensorData wrappers around the runtime
// buffers we receive in DoFft.
class MetalFftPlan : public fft::Plan {
 public:
  MetalFftPlan(MPSGraphExecutable* executable, fft::Type type, int rank,
               int batch, MPSDataType input_dtype, MPSDataType output_dtype,
               std::vector<NSInteger> input_shape,
               std::vector<NSInteger> output_shape)
      : executable_(executable),
        type_(type),
        rank_(rank),
        batch_(batch),
        input_dtype_(input_dtype),
        output_dtype_(output_dtype),
        input_shape_(std::move(input_shape)),
        output_shape_(std::move(output_shape)) {}

  MPSGraphExecutable* executable() const { return executable_; }
  fft::Type type() const { return type_; }
  int rank() const { return rank_; }
  int batch() const { return batch_; }
  MPSDataType input_dtype() const { return input_dtype_; }
  MPSDataType output_dtype() const { return output_dtype_; }
  NSArray<NSNumber*>* input_shape() const {
    return ToNSArray(input_shape_);
  }
  NSArray<NSNumber*>* output_shape() const {
    return ToNSArray(output_shape_);
  }

  // MPSGraph with scalingMode=Size produces the already-normalised result,
  // so FftThunk doesn't need a BLAS post-scale. See fft::Plan docstring.
  uint64_t PostFftScaleFactor(fft::Type /*type*/,
                              uint64_t /*output_distance*/) const override {
    return 1;
  }

 private:
  static NSArray<NSNumber*>* ToNSArray(const std::vector<NSInteger>& v) {
    NSMutableArray<NSNumber*>* out =
        [NSMutableArray arrayWithCapacity:v.size()];
    for (NSInteger d : v) [out addObject:@(d)];
    return out;
  }

  MPSGraphExecutable* executable_;
  fft::Type type_;
  int rank_;
  int batch_;
  MPSDataType input_dtype_;
  MPSDataType output_dtype_;
  std::vector<NSInteger> input_shape_;
  std::vector<NSInteger> output_shape_;
};

// Wraps a DeviceAddressBase backed by MetalAllocator into an
// MPSGraphTensorData with the requested shape + dtype. The BFC allocator
// hands us sub-slices of larger MTLBuffers, so we route through MPSNDArray
// (which accepts an offset) instead of MPSGraphTensorData's direct
// MTLBuffer-with-shape initialiser (no offset support).
absl::StatusOr<MPSGraphTensorData*> MakeTensorData(MetalAllocator* allocator,
                                                    const void* ptr,
                                                    NSArray<NSNumber*>* shape,
                                                    MPSDataType dtype) {
  auto resolved = allocator->Resolve(ptr);
  if (!resolved.has_value()) {
    return absl::InvalidArgumentError(
        "MetalFft: input/output pointer not owned by the executor's allocator");
  }
  MPSNDArrayDescriptor* descriptor =
      [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:shape];
  if (descriptor == nil) {
    return absl::InternalError("MetalFft: MPSNDArrayDescriptor init returned nil");
  }
  MPSNDArray* ndarray =
      [[MPSNDArray alloc] initWithBuffer:resolved->buffer
                                  offset:resolved->offset
                              descriptor:descriptor];
  if (ndarray == nil) {
    return absl::InternalError("MetalFft: MPSNDArray init returned nil");
  }
  MPSGraphTensorData* data =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray];
  if (data == nil) {
    return absl::InternalError("MetalFft: MPSGraphTensorData init returned nil");
  }
  return data;
}

}  // namespace

std::unique_ptr<fft::Plan> MetalFft::CreateBatchedPlanWithScratchAllocator(
    Stream* stream, int rank, uint64_t* elem_count, uint64_t* input_embed,
    uint64_t input_stride, uint64_t input_distance, uint64_t* output_embed,
    uint64_t output_stride, uint64_t output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator* /*scratch*/) {
  if (rank < 1 || rank > 3) {
    LOG(ERROR) << "MetalFft: unsupported rank " << rank;
    return nullptr;
  }
  if (in_place_fft) {
    LOG(ERROR) << "MetalFft: in-place FFT is not supported";
    return nullptr;
  }
  if (input_stride != 1 || output_stride != 1) {
    LOG(ERROR) << "MetalFft: only unit-strided dense layouts are supported";
    return nullptr;
  }
  // Reject any layout where the storage isn't densely packed for the same
  // reason MPSGraph FFT only takes axis lists on dense tensors.
  if (input_distance != Prod(input_embed, rank) ||
      output_distance != Prod(output_embed, rank)) {
    LOG(ERROR) << "MetalFft: non-dense (embed != distance) layouts unsupported";
    return nullptr;
  }
  std::optional<FftCase> info = Classify(type);
  if (!info.has_value()) {
    LOG(ERROR) << "MetalFft: F64 variants are unsupported on Apple silicon";
    return nullptr;
  }

  @autoreleasepool {
    // Build the graph, the FFT op, and the input placeholder.
    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSGraphFFTDescriptor* descriptor =
        [MPSGraphFFTDescriptor descriptor];
    descriptor.inverse = info->inverse;
    // scalingMode=Size: MPSGraph divides inverse output by N internally so
    // PostFftScaleFactor=1 above is honest.
    descriptor.scalingMode = info->inverse ? MPSGraphFFTScalingModeSize
                                            : MPSGraphFFTScalingModeNone;
    // For C2R / R2C round-to-odd, FftThunk passes us the real-side length in
    // fft_length[rank-1] and the Hermitean-side length in the complex-side
    // embed. Round-to-odd is "true" iff the real length is odd.
    descriptor.roundToOddHermitean = (elem_count[rank - 1] & 1) == 1;

    // Input dtype + shape. For R2C input is real, output is Hermitean
    // complex of length N/2+1. For C2R input is Hermitean complex, output is
    // real of length elem_count[rank-1].
    MPSDataType input_dtype = info->real_input ? MPSDataTypeFloat32
                                                : MPSDataTypeComplexFloat32;
    MPSDataType output_dtype = info->real_output ? MPSDataTypeFloat32
                                                  : MPSDataTypeComplexFloat32;
    NSArray<NSNumber*>* input_shape =
        MakeShape(batch_count, input_embed, rank);
    NSArray<NSNumber*>* output_shape =
        MakeShape(batch_count, output_embed, rank);
    NSArray<NSNumber*>* axes = MakeAxes(rank);

    MPSGraphTensor* input_placeholder =
        [graph placeholderWithShape:input_shape
                           dataType:input_dtype
                               name:@"input"];
    MPSGraphTensor* result;
    if (info->real_to_hermitean) {
      result = [graph realToHermiteanFFTWithTensor:input_placeholder
                                             axes:axes
                                       descriptor:descriptor
                                             name:@"r2c"];
    } else if (info->hermitean_to_real) {
      result = [graph HermiteanToRealFFTWithTensor:input_placeholder
                                             axes:axes
                                       descriptor:descriptor
                                             name:@"c2r"];
    } else {
      result = [graph fastFourierTransformWithTensor:input_placeholder
                                                axes:axes
                                          descriptor:descriptor
                                                name:@"c2c"];
    }

    // Compile. `feedShape` is needed by the executable but its shape comes
    // directly from the placeholder. `targetTensors` lists the outputs.
    MPSGraphShapedType* feed_type =
        [[MPSGraphShapedType alloc] initWithShape:input_shape
                                          dataType:input_dtype];
    NSDictionary* feeds = @{input_placeholder : feed_type};
    NSArray<MPSGraphTensor*>* targets = @[ result ];
    NSError* err = nil;
    MPSGraphCompilationDescriptor* compile_desc =
        [[MPSGraphCompilationDescriptor alloc] init];
    MPSGraphExecutable* executable =
        [graph compileWithDevice:nil
                            feeds:feeds
                    targetTensors:targets
                 targetOperations:nil
            compilationDescriptor:compile_desc];
    if (executable == nil) {
      LOG(ERROR) << "MetalFft: MPSGraph compile returned nil"
                 << (err != nil ? std::string(" err=") +
                                      [[err description] UTF8String]
                                : std::string());
      return nullptr;
    }

    std::vector<NSInteger> input_shape_vec;
    input_shape_vec.reserve(input_shape.count);
    for (NSNumber* d in input_shape)
      input_shape_vec.push_back([d integerValue]);
    std::vector<NSInteger> output_shape_vec;
    output_shape_vec.reserve(output_shape.count);
    for (NSNumber* d in output_shape)
      output_shape_vec.push_back([d integerValue]);

    return std::make_unique<MetalFftPlan>(
        executable, type, rank, batch_count, input_dtype, output_dtype,
        std::move(input_shape_vec), std::move(output_shape_vec));
  }
}

void MetalFft::UpdatePlanWithScratchAllocator(Stream* /*stream*/,
                                                fft::Plan* /*plan*/,
                                                ScratchAllocator* /*scratch*/) {
  // MPSGraph manages its own internal buffers; nothing to update.
}

namespace {

// Common encode path used by every DoFft overload below. Resolves the
// input/output device pointers to MTLBuffers, wraps them as MPSGraphTensorData,
// and encodes the executable onto a fresh command buffer via MetalStream's
// EncodeWithCommandBuffer helper (which handles labelling, error tracking,
// and tail-buffer maintenance).
bool RunPlan(MetalFft* fft, Stream* stream, fft::Plan* plan,
             const void* input_ptr, void* output_ptr) {
  auto* metal_stream = static_cast<MetalStream*>(stream);
  auto* metal_plan = static_cast<MetalFftPlan*>(plan);
  MetalAllocator* allocator = fft->executor()->allocator();

  @autoreleasepool {
    absl::StatusOr<MPSGraphTensorData*> input_data =
        MakeTensorData(allocator, input_ptr, metal_plan->input_shape(),
                       metal_plan->input_dtype());
    if (!input_data.ok()) {
      LOG(ERROR) << "MetalFft::DoFft: " << input_data.status();
      return false;
    }
    absl::StatusOr<MPSGraphTensorData*> output_data =
        MakeTensorData(allocator, output_ptr, metal_plan->output_shape(),
                       metal_plan->output_dtype());
    if (!output_data.ok()) {
      LOG(ERROR) << "MetalFft::DoFft: " << output_data.status();
      return false;
    }

    NSArray* inputs = @[ *input_data ];
    NSArray* outputs = @[ *output_data ];
    MPSGraphExecutable* exe = metal_plan->executable();

    absl::StatusOr<id<MTLCommandBuffer>> encoded =
        metal_stream->EncodeWithCommandBuffer(
            /*label=*/"MetalFft",
            /*op_name=*/"MetalFft::DoFft",
            [&](id<MTLCommandBuffer> cmd_buf)
                -> absl::StatusOr<id<MTLCommandBuffer>> {
              MPSCommandBuffer* mps_cmd_buf =
                  [MPSCommandBuffer commandBufferWithCommandBuffer:cmd_buf];
              MPSGraphExecutableExecutionDescriptor* exec_desc =
                  [[MPSGraphExecutableExecutionDescriptor alloc] init];
              [exe encodeToCommandBuffer:mps_cmd_buf
                            inputsArray:inputs
                           resultsArray:outputs
                    executionDescriptor:exec_desc];
              // MPSGraph may have called commitAndContinue internally; the
              // final underlying buffer is what we must track + commit. The
              // stream commits it after installing the error handler.
              return [mps_cmd_buf commandBuffer];
            });
    if (!encoded.ok()) {
      LOG(ERROR) << "MetalFft::DoFft: " << encoded.status();
      return false;
    }
    return true;
  }
}

}  // namespace

bool MetalFft::DoFft(Stream* stream, fft::Plan* plan,
                      const DeviceAddress<std::complex<float>>& input,
                      DeviceAddress<std::complex<float>>* output) {
  return RunPlan(this, stream, plan, input.opaque(), output->opaque());
}

bool MetalFft::DoFft(Stream* /*stream*/, fft::Plan* /*plan*/,
                      const DeviceAddress<std::complex<double>>& /*input*/,
                      DeviceAddress<std::complex<double>>* /*output*/) {
  LOG(ERROR) << "MetalFft: complex<double> FFT is not supported";
  return false;
}

bool MetalFft::DoFft(Stream* stream, fft::Plan* plan,
                      const DeviceAddress<float>& input,
                      DeviceAddress<std::complex<float>>* output) {
  return RunPlan(this, stream, plan, input.opaque(), output->opaque());
}

bool MetalFft::DoFft(Stream* /*stream*/, fft::Plan* /*plan*/,
                      const DeviceAddress<double>& /*input*/,
                      DeviceAddress<std::complex<double>>* /*output*/) {
  LOG(ERROR) << "MetalFft: double R2C FFT is not supported";
  return false;
}

bool MetalFft::DoFft(Stream* stream, fft::Plan* plan,
                      const DeviceAddress<std::complex<float>>& input,
                      DeviceAddress<float>* output) {
  return RunPlan(this, stream, plan, input.opaque(), output->opaque());
}

bool MetalFft::DoFft(Stream* /*stream*/, fft::Plan* /*plan*/,
                      const DeviceAddress<std::complex<double>>& /*input*/,
                      DeviceAddress<double>* /*output*/) {
  LOG(ERROR) << "MetalFft: complex<double> C2R FFT is not supported";
  return false;
}

}  // namespace stream_executor::metal

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_metalfft, {
  auto* registry = stream_executor::PluginRegistry::Instance();
  absl::Status status =
      registry->RegisterFactory<stream_executor::PluginRegistry::FftFactory>(
          stream_executor::metal::kMetalPlatformId, "MPSGraphFFT",
          [](stream_executor::StreamExecutor* parent)
              -> stream_executor::fft::FftSupport* {
            return new stream_executor::metal::MetalFft(
                static_cast<stream_executor::metal::MetalExecutor*>(parent));
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register MetalFft factory: " << status.message();
  }
});
