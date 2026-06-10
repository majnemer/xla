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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_GRAPH_EXECUTABLE_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_GRAPH_EXECUTABLE_H_

namespace stream_executor {
namespace metal {

// Opaque holder for an MPSGraphExecutable. C++-only header so it can be
// included from non-Obj-C translation units (xla/backends/metal/*.cc). The
// executable is retained on Wrap() and released on destruction. Mirrors
// PipelineStateRef (metal_pso_probe.h).
class GraphExecutableRef {
 public:
  GraphExecutableRef();
  ~GraphExecutableRef();

  GraphExecutableRef(const GraphExecutableRef&) = delete;
  GraphExecutableRef& operator=(const GraphExecutableRef&) = delete;

  GraphExecutableRef(GraphExecutableRef&& other) noexcept;
  GraphExecutableRef& operator=(GraphExecutableRef&& other) noexcept;

  // Wraps and retains an MPSGraphExecutable passed as an opaque pointer
  // (`(__bridge void*)executable` at the call site).
  static GraphExecutableRef Wrap(void* executable);

  // The underlying MPSGraphExecutable as a void*. Cast inside .mm consumers
  // via `(__bridge MPSGraphExecutable*)ref.opaque()`.
  void* opaque() const { return opaque_; }
  bool empty() const { return opaque_ == nullptr; }

 private:
  void* opaque_;  // CFRetained MPSGraphExecutable (treat as +1).
};

}  // namespace metal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_GRAPH_EXECUTABLE_H_
