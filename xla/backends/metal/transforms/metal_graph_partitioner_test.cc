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

#include "xla/backends/metal/transforms/metal_graph_partitioner.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace metal {
namespace {

class MetalGraphPartitionerTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunPartitioner(HloModule* module,
                                      bool supports_bf16 = false) {
    MetalGraphCapabilities caps;
    caps.supports_bf16 = supports_bf16;
    MetalGraphPartitioner pass(caps);
    return RunHloPass(&pass, module);
  }

  // Returns the unique __metal_graph fusion in the entry computation, or
  // nullptr.
  static HloInstruction* FindGraphFusion(HloModule* module) {
    HloInstruction* found = nullptr;
    for (HloInstruction* instr :
         module->entry_computation()->instructions()) {
      if (IsMetalGraphFusion(*instr)) {
        EXPECT_EQ(found, nullptr) << "multiple graph fusions";
        found = instr;
      }
    }
    return found;
  }

  static std::vector<HloOpcode> FusedOpcodes(const HloInstruction* fusion) {
    std::vector<HloOpcode> opcodes;
    for (const HloInstruction* instr :
         fusion->fused_instructions_computation()->instructions()) {
      if (instr->opcode() != HloOpcode::kParameter) {
        opcodes.push_back(instr->opcode());
      }
    }
    return opcodes;
  }
};

TEST_F(MetalGraphPartitionerTest, SoloDotIsCaptured) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,8] parameter(1)
      ROOT d = f32[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                      rhs_contracting_dims={0}
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion, module->entry_computation()->root_instruction());
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(HloOpcode::kDot));
}

TEST_F(MetalGraphPartitionerTest, DotTanhDotIsOneRegion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      p2 = f32[8,8] parameter(2)
      d0 = f32[8,8] dot(p0, p1), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
      t = f32[8,8] tanh(d0)
      ROOT d1 = f32[8,8] dot(t, p2), lhs_contracting_dims={1},
                                     rhs_contracting_dims={0}
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(
                  HloOpcode::kDot, HloOpcode::kTanh, HloOpcode::kDot));
}

TEST_F(MetalGraphPartitionerTest, ReconvergentDiamondIsAbsorbed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      d = f32[8,8] dot(p0, p1), lhs_contracting_dims={1},
                                rhs_contracting_dims={0}
      t = f32[8,8] tanh(d)
      ROOT m = f32[8,8] multiply(d, t)
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(
                  HloOpcode::kDot, HloOpcode::kTanh, HloOpcode::kMultiply));
}

TEST_F(MetalGraphPartitionerTest, GenuineFanOutStopsAtDot) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      d = f32[8,8] dot(p0, p1), lhs_contracting_dims={1},
                                rhs_contracting_dims={0}
      t = f32[8,8] tanh(d)
      x = f32[8,8] exponential(d)
      ROOT tup = (f32[8,8], f32[8,8]) tuple(t, x)
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(HloOpcode::kDot));
}

TEST_F(MetalGraphPartitionerTest, BiasEpilogueAbsorbsBroadcast) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,8] parameter(1)
      bias = f32[8] parameter(2)
      d = f32[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
      b = f32[16,8] broadcast(bias), dimensions={1}
      ROOT a = f32[16,8] add(d, b)
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(
                  HloOpcode::kDot, HloOpcode::kBroadcast, HloOpcode::kAdd));
  EXPECT_EQ(fusion->operand_count(), 3);
}

TEST_F(MetalGraphPartitionerTest, GrowsThroughMonoidReduce) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add_comp {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT s = f32[] add(a, b)
    }
    ENTRY e {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,8] parameter(1)
      d = f32[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
      x = f32[16,8] exponential(d)
      c0 = f32[] constant(0)
      ROOT r = f32[16] reduce(x, c0), dimensions={1}, to_apply=add_comp
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(
                  HloOpcode::kDot, HloOpcode::kExp, HloOpcode::kConstant,
                  HloOpcode::kReduce));
}

TEST_F(MetalGraphPartitionerTest, CustomCombinerReduceStopsGrowth) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    weird_comp {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      s = f32[] add(a, b)
      ROOT t = f32[] tanh(s)
    }
    ENTRY e {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,8] parameter(1)
      d = f32[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
      x = f32[16,8] exponential(d)
      c0 = f32[] constant(0)
      ROOT r = f32[16] reduce(x, c0), dimensions={1}, to_apply=weird_comp
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(HloOpcode::kDot, HloOpcode::kExp));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduce);
}

TEST_F(MetalGraphPartitionerTest, StrandedSingleUseProducerBlocksAbsorption) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[16,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      d = f32[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
      io = f32[16,8] iota(), iota_dimension=0
      ROOT m = f32[16,8] multiply(d, io)
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  // Absorbing the multiply would strand the single-use iota chain that fuses
  // with it today.
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(HloOpcode::kDot));
}

TEST_F(MetalGraphPartitionerTest, MultiUseProducerStaysOutside) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      t = f32[8,8] tanh(p0)
      d = f32[8,8] dot(t, p1), lhs_contracting_dims={1},
                               rhs_contracting_dims={0}
      ROOT tup = (f32[8,8], f32[8,8]) tuple(d, t)
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* fusion = FindGraphFusion(module.get());
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(FusedOpcodes(fusion),
              ::testing::UnorderedElementsAre(HloOpcode::kDot));
  // tanh kept its other user; no duplication into the region.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 5);
}

TEST_F(MetalGraphPartitionerTest, UntranslatableDotIsLeftAlone) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = c64[16,32] parameter(0)
      p1 = c64[32,8] parameter(1)
      ROOT d = c64[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                      rhs_contracting_dims={0}
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindGraphFusion(module.get()), nullptr);
}

TEST_F(MetalGraphPartitionerTest, Bf16DotGatedOnCapability) {
  constexpr absl::string_view kHlo = R"(
    ENTRY e {
      p0 = bf16[16,32] parameter(0)
      p1 = bf16[32,8] parameter(1)
      ROOT d = bf16[16,8] dot(p0, p1), lhs_contracting_dims={1},
                                       rhs_contracting_dims={0}
    })";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed, RunPartitioner(module.get(), /*supports_bf16=*/false));
    EXPECT_FALSE(changed);
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed, RunPartitioner(module.get(), /*supports_bf16=*/true));
    EXPECT_TRUE(changed);
    EXPECT_NE(FindGraphFusion(module.get()), nullptr);
  }
}

TEST_F(MetalGraphPartitionerTest, NoAnchorNoChange) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add_comp {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT s = f32[] add(a, b)
    }
    ENTRY e {
      p0 = f32[16,8] parameter(0)
      x = f32[16,8] exponential(p0)
      c0 = f32[] constant(0)
      ROOT r = f32[16] reduce(x, c0), dimensions={1}, to_apply=add_comp
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(MetalGraphPartitionerTest, CapturesInsideWhileBody) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    body {
      p = (f32[8,8], f32[8,8]) parameter(0)
      a = f32[8,8] get-tuple-element(p), index=0
      b = f32[8,8] get-tuple-element(p), index=1
      d = f32[8,8] dot(a, b), lhs_contracting_dims={1},
                              rhs_contracting_dims={0}
      ROOT t = (f32[8,8], f32[8,8]) tuple(d, b)
    }
    cond {
      p = (f32[8,8], f32[8,8]) parameter(0)
      ROOT c = pred[] constant(false)
    }
    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      t0 = (f32[8,8], f32[8,8]) tuple(p0, p1)
      ROOT w = (f32[8,8], f32[8,8]) while(t0), condition=cond, body=body
    })"));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPartitioner(module.get()));
  EXPECT_TRUE(changed);
  // The dot inside the while body is captured.
  const HloComputation* body = module->GetComputationWithName("body");
  ASSERT_NE(body, nullptr);
  bool found = false;
  for (const HloInstruction* instr : body->instructions()) {
    found |= IsMetalGraphFusion(*instr);
  }
  EXPECT_TRUE(found);
}

}  // namespace
}  // namespace metal
}  // namespace xla
