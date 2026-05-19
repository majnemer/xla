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

// Registers a stub collectives implementation for the Metal platform.
//
// GpuExecutable::ExecuteAsyncOnStream calls CollectiveParams::Create, which
// calls GpuCollectives::Default(platform_name). With no registration the
// lookup fails and Default CHECK_OK-crashes. Metal has no collective
// communication backend today (single-device only), so we register the
// existing GpuCollectivesStub — its IsImplemented() returns false and every
// collective op surfaces a clean Unimplemented status. This keeps single-
// device execution working without pretending we support collectives.

#include <memory>

#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/core/collectives/collectives_registry.h"

XLA_COLLECTIVES_REGISTER("METAL", "stub", /*priority=*/0,
                         std::make_unique<xla::gpu::GpuCollectivesStub>());
