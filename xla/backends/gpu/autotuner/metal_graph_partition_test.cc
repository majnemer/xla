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

#include "xla/backends/gpu/autotuner/metal_graph_partition.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::UnorderedElementsAre;

constexpr absl::string_view kDotTanhDot = R"hlo(
HloModule m
fused {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[4,4] parameter(2)
  d0 = f32[4,4] dot(p0, p1), lhs_contracting_dims={1},
                             rhs_contracting_dims={0}
  t = f32[4,4] tanh(d0)
  ROOT d1 = f32[4,4] dot(t, p2), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
}
ENTRY e {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[4,4] parameter(2)
  ROOT f = f32[4,4] fusion(p0, p1, p2), kind=kCustom, calls=fused,
      backend_config={"fusion_backend_config":{"kind":"__metal_graph"}}
}
)hlo";

class MetalGraphPartitionBackendTest : public ::testing::Test {
 protected:
  static se::StreamExecutor* GetExecutor() {
    auto platform = se::PlatformManager::PlatformWithName("METAL");
    if (!platform.ok() || (*platform)->VisibleDeviceCount() < 1) {
      return nullptr;
    }
    return (*platform)->ExecutorForDevice(0).value_or(nullptr);
  }

  MetalGraphPartitionBackendTest()
      : executor_(GetExecutor()),
        target_config_(executor_),
        backend_(executor_, &debug_options_, /*compiler=*/nullptr,
                 &target_config_) {}

  se::StreamExecutor* executor_;
  DebugOptions debug_options_;
  Compiler::GpuTargetConfig target_config_;
  MetalGraphPartitionBackend backend_;
};

std::vector<std::string> GrownAnchors(const autotuner::BackendConfig& config) {
  return std::vector<std::string>(
      config.metal_graph_partition().grown_anchors().begin(),
      config.metal_graph_partition().grown_anchors().end());
}

// Returns (graph fusion body op counts, bare non-param op count) for the
// entry computation.
struct EntryShape {
  std::vector<int> fusion_body_ops;
  int bare_ops = 0;
  bool all_marked = true;
};
EntryShape SummarizeEntry(const HloModule& module) {
  EntryShape summary;
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (metal::IsMetalGraphFusion(*instr)) {
      int ops = 0;
      for (const HloInstruction* body_instr :
           Cast<HloFusionInstruction>(instr)
               ->fused_instructions_computation()
               ->instructions()) {
        ops += body_instr->opcode() != HloOpcode::kParameter ? 1 : 0;
      }
      summary.fusion_body_ops.push_back(ops);
      auto config = instr->backend_config<GpuBackendConfig>();
      summary.all_marked &=
          config.ok() &&
          config->fusion_backend_config().has_metal_graph_fusion_config();
    } else {
      summary.bare_ops++;
    }
  }
  return summary;
}

TEST_F(MetalGraphPartitionBackendTest, EnumeratesPlansForTwoAnchors) {
  if (executor_ == nullptr) GTEST_SKIP() << "No Metal device.";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDotTanhDot));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto configs, backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 4);
  EXPECT_THAT(GrownAnchors(*configs[0]), UnorderedElementsAre());
  EXPECT_THAT(GrownAnchors(*configs[1]), UnorderedElementsAre("d0"));
  EXPECT_THAT(GrownAnchors(*configs[2]), UnorderedElementsAre("d1"));
  EXPECT_THAT(GrownAnchors(*configs[3]), UnorderedElementsAre("d0", "d1"));
}

TEST_F(MetalGraphPartitionBackendTest, SoloAnchorFusionHasNoPlans) {
  if (executor_ == nullptr) GTEST_SKIP() << "No Metal device.";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"hlo(
    HloModule m
    fused {
      p0 = f32[4,4] parameter(0)
      p1 = f32[4,4] parameter(1)
      ROOT d = f32[4,4] dot(p0, p1), lhs_contracting_dims={1},
                                     rhs_contracting_dims={0}
    }
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[4,4] parameter(1)
      ROOT f = f32[4,4] fusion(p0, p1), kind=kCustom, calls=fused,
          backend_config={"fusion_backend_config":{"kind":"__metal_graph"}}
    }
  )hlo"));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto configs, backend_.GetSupportedConfigs(*fusion));
  EXPECT_TRUE(configs.empty());
}

TEST_F(MetalGraphPartitionBackendTest, SingleAnchorEpilogueOffersSplitOnly) {
  if (executor_ == nullptr) GTEST_SKIP() << "No Metal device.";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"hlo(
    HloModule m
    fused {
      p0 = f32[4,4] parameter(0)
      p1 = f32[4,4] parameter(1)
      d = f32[4,4] dot(p0, p1), lhs_contracting_dims={1},
                                rhs_contracting_dims={0}
      ROOT t = f32[4,4] tanh(d)
    }
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[4,4] parameter(1)
      ROOT f = f32[4,4] fusion(p0, p1), kind=kCustom, calls=fused,
          backend_config={"fusion_backend_config":{"kind":"__metal_graph"}}
    }
  )hlo"));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto configs, backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_TRUE(GrownAnchors(*configs[0]).empty());

  // Applying the split: solo dot capture, tanh back to native.
  TF_ASSERT_OK(backend_.ApplyConfig(*fusion, *configs[0]));
  EntryShape summary = SummarizeEntry(*module);
  EXPECT_THAT(summary.fusion_body_ops, UnorderedElementsAre(1));
  EXPECT_EQ(summary.bare_ops, 1);
  EXPECT_TRUE(summary.all_marked);
}

TEST_F(MetalGraphPartitionBackendTest, AppliesGrownInnerAnchorPlan) {
  if (executor_ == nullptr) GTEST_SKIP() << "No Metal device.";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDotTanhDot));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  autotuner::BackendConfig config;
  config.mutable_metal_graph_partition()->add_grown_anchors("d0");
  TF_ASSERT_OK(backend_.ApplyConfig(*fusion, config));

  // d0 grows over tanh (d1 is excluded from its region); d1 becomes a solo
  // capture; nothing returns to native.
  EntryShape summary = SummarizeEntry(*module);
  EXPECT_THAT(summary.fusion_body_ops, UnorderedElementsAre(2, 1));
  EXPECT_EQ(summary.bare_ops, 0);
  EXPECT_TRUE(summary.all_marked);
}

TEST_F(MetalGraphPartitionBackendTest, AppliesAllSoloPlan) {
  if (executor_ == nullptr) GTEST_SKIP() << "No Metal device.";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDotTanhDot));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  autotuner::BackendConfig config;
  config.mutable_metal_graph_partition();
  TF_ASSERT_OK(backend_.ApplyConfig(*fusion, config));

  // Two solo captures; tanh returns to native.
  EntryShape summary = SummarizeEntry(*module);
  EXPECT_THAT(summary.fusion_body_ops, UnorderedElementsAre(1, 1));
  EXPECT_EQ(summary.bare_ops, 1);
  EXPECT_TRUE(summary.all_marked);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
