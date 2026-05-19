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

#ifndef XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_SOURCE_H_
#define XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_SOURCE_H_

#include <string>
#include <utility>

#include "xla/codegen/kernel_source.h"

namespace xla {
namespace metal {

// MSL source text plus the entry-point function name within it.
class MslKernelSource final : public KernelSource {
 public:
  MslKernelSource(std::string source, std::string entry_point)
      : source_(std::move(source)), entry_point_(std::move(entry_point)) {}

  MslKernelSource(MslKernelSource&&) = default;
  MslKernelSource& operator=(MslKernelSource&&) = default;

  const std::string& source() const { return source_; }
  const std::string& entry_point() const { return entry_point_; }

  std::string ToString() const final { return source_; }

 private:
  std::string source_;
  std::string entry_point_;
};

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_SOURCE_H_
