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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/metal/metal_compute_capability.pb.h"

namespace stream_executor {

// Apple Silicon (Metal) compute capability, as reported by the device
// description. Pure-C++ — no <Metal/Metal.h> in this header so it can be
// included anywhere in XLA without forcing Obj-C++ compilation.
//
// At the runtime boundary (xla/stream_executor/metal/), MTLGPUFamily values
// are mapped into the integer family_generation here.
class MetalComputeCapability {
 public:
  MetalComputeCapability() = default;

  // family_generation: the N in MTLGPUFamilyApple<N>.
  //   M1 = 7, M2 = 8, M3 = 9, M4 = 10. 0 means unknown.
  // metal_3_supported: whether [device supportsFamily:MTLGPUFamilyMetal3]
  //   returns true. Required for the Metal 3 language version and the richer
  //   simdgroup_matrix intrinsics; depends on both hardware (Apple7+) and the
  //   macOS version (13+).
  explicit MetalComputeCapability(int family_generation, bool metal_3_supported)
      : family_generation_(family_generation),
        metal_3_supported_(metal_3_supported) {}

  explicit MetalComputeCapability(const MetalComputeCapabilityProto& proto)
      : MetalComputeCapability(proto.family_generation(),
                               proto.metal_3_supported()) {}

  int family_generation() const { return family_generation_; }
  bool metal_3_supported() const { return metal_3_supported_; }

  // Apple9+ (M3+) has hardware bf16 in SIMDgroup and matrix instructions.
  // Older Apple Silicon falls back to fp32 via FloatNormalization.
  bool has_bf16() const { return family_generation_ >= 9; }

  // simdgroup_matrix types are available on Apple7+; the most useful matrix
  // intrinsics (matmul-shaped accumulate) require Metal 3.
  bool has_simdgroup_matrix() const { return family_generation_ >= 7; }

  std::string ToString() const {
    return absl::StrCat("Apple", family_generation_,
                        metal_3_supported_ ? " (Metal 3)" : " (Metal 2.4)");
  }

  MetalComputeCapabilityProto ToProto() const {
    MetalComputeCapabilityProto proto;
    proto.set_family_generation(family_generation_);
    proto.set_metal_3_supported(metal_3_supported_);
    return proto;
  }

  static MetalComputeCapability FromProto(
      const MetalComputeCapabilityProto& proto) {
    return MetalComputeCapability(proto.family_generation(),
                                  proto.metal_3_supported());
  }

  bool operator==(const MetalComputeCapability& other) const {
    return family_generation_ == other.family_generation_ &&
           metal_3_supported_ == other.metal_3_supported_;
  }

  bool operator!=(const MetalComputeCapability& other) const {
    return !(*this == other);
  }

 private:
  int family_generation_ = 0;
  bool metal_3_supported_ = false;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_
