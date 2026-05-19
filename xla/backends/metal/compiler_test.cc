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

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace metal {
namespace {

// Compile-time registration: pulling in the compiler library should make
// Compiler::GetForPlatform succeed for the Metal platform id.
TEST(MetalCompilerTest, RegisteredUnderMetalPlatformId) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_executor::metal::kMetalPlatformId));
  ASSERT_NE(compiler, nullptr);
  EXPECT_EQ(compiler->PlatformId(),
            stream_executor::metal::kMetalPlatformId);
}

}  // namespace
}  // namespace metal
}  // namespace xla
