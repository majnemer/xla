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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_PSO_PROBE_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_PSO_PROBE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
namespace metal {

// Opaque holder for an id<MTLComputePipelineState>. C++-only header so it can
// be included from non-Obj-C translation units (xla/backends/metal/*.cc).
// The PSO is retained on construction and released on destruction.
class PipelineStateRef {
 public:
  PipelineStateRef();
  ~PipelineStateRef();

  PipelineStateRef(const PipelineStateRef&) = delete;
  PipelineStateRef& operator=(const PipelineStateRef&) = delete;

  PipelineStateRef(PipelineStateRef&& other) noexcept;
  PipelineStateRef& operator=(PipelineStateRef&& other) noexcept;

  // Returns the underlying id<MTLComputePipelineState> as a void* so this
  // header stays Obj-C-free. Cast inside .mm consumers via
  // `(__bridge id<MTLComputePipelineState>)ref.opaque()`.
  void* opaque() const { return opaque_; }
  bool empty() const { return opaque_ == nullptr; }

 private:
  friend class PipelineStateBuilder;
  void* opaque_;  // CFRetained id<MTLComputePipelineState> (treat as +1).
};

// Result of compiling MSL into a PSO. `pso_max_threads` is what the driver
// reports as [pso maxTotalThreadsPerThreadgroup] — the *hard* per-launch
// ceiling for this kernel, possibly below the launch's intended thread count
// when the kernel is register-heavy.
struct CompiledPipeline {
  std::shared_ptr<PipelineStateRef> pso;
  int64_t pso_max_threads = 0;
};

// Compiles `msl` on `device` (passed as id<MTLDevice> via an opaque void*),
// resolves the entry point `entry_name`, and builds an MTLComputePipelineState
// with descriptor.maxTotalThreadsPerThreadgroup = `descriptor_thread_hint`.
// The descriptor hint is a register-budget signal to the compiler ("you won't
// be asked to host more than N threads"). Apple's docs note that the resulting
// PSO may still report a lower ceiling if even at that hint the kernel exceeds
// register capacity — the caller must read pso_max_threads and decide whether
// to retry at a smaller hint.
//
// Returns the PSO + its self-reported max-threads-per-threadgroup, or an
// error describing which compilation stage failed.
absl::StatusOr<CompiledPipeline> CompileAndProbe(
    void* device, absl::string_view msl, absl::string_view entry_name,
    int64_t descriptor_thread_hint);

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_PSO_PROBE_H_
