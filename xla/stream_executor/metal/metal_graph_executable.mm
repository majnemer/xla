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

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "xla/stream_executor/metal/metal_graph_executable.h"

namespace stream_executor {
namespace metal {

GraphExecutableRef::GraphExecutableRef() : opaque_(nullptr) {}

GraphExecutableRef::~GraphExecutableRef() {
  if (opaque_ != nullptr) {
    MPSGraphExecutable* executable =
        (__bridge_transfer MPSGraphExecutable*)opaque_;
    (void)executable;  // ARC releases on scope exit.
    opaque_ = nullptr;
  }
}

GraphExecutableRef::GraphExecutableRef(GraphExecutableRef&& other) noexcept
    : opaque_(other.opaque_) {
  other.opaque_ = nullptr;
}

GraphExecutableRef& GraphExecutableRef::operator=(
    GraphExecutableRef&& other) noexcept {
  if (this != &other) {
    if (opaque_ != nullptr) {
      MPSGraphExecutable* executable =
          (__bridge_transfer MPSGraphExecutable*)opaque_;
      (void)executable;
    }
    opaque_ = other.opaque_;
    other.opaque_ = nullptr;
  }
  return *this;
}

GraphExecutableRef GraphExecutableRef::Wrap(void* executable) {
  GraphExecutableRef ref;
  if (executable != nullptr) {
    MPSGraphExecutable* obj = (__bridge MPSGraphExecutable*)executable;
    ref.opaque_ = (__bridge_retained void*)obj;
  }
  return ref;
}

}  // namespace metal
}  // namespace stream_executor
