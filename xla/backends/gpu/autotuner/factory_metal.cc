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

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Metal has no cuDNN / cuBLAS / Triton-style backends to autotune over —
// the kFusion path goes straight through MlirKernelEmitter to MSL. Returning
// an empty vector keeps the autotuner pass a no-op rather than crashing on
// a missing platform registration. Matches the GetCodegenBackends signature
// for symmetry with the CUDA / ROCm factories.
std::vector<std::unique_ptr<CodegenBackend>> GetCodegenBackendsForMetal(
    stream_executor::StreamExecutor* /*stream_executor*/,
    stream_executor::DeviceAddressAllocator* /*device_allocator*/,
    const DebugOptions* /*debug_options*/, Compiler* /*compiler*/,
    const Compiler::GpuTargetConfig* /*target_config*/,
    const AliasInfo* /*alias_info*/, mlir::MLIRContext* /*mlir_context*/,
    absl::Span<const autotuner::Backend> /*backend_allowlist*/) {
  return {};
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(
    GetCodegenBackendsMetalRegistration, GetCodegenBackends,
    stream_executor::metal::kMetalPlatformId, GetCodegenBackendsForMetal);

}  // namespace gpu
}  // namespace xla
