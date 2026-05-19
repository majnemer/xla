// RUN: emitters_opt %s -xla-metal-expand-float-ops -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @log1p_f32(
// CHECK-SAME:    %[[X:.*]]: f32
func.func @log1p_f32(%x: f32) -> f32 {
  %r = math.log1p %x : f32
  return %r : f32
}
// CHECK-NOT:  math.log1p
// CHECK:      %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:      %[[X_PLUS_ONE:.*]] = arith.addf %[[X]], %[[ONE]] : f32
// CHECK:      %[[LARGE:.*]] = math.log %[[X_PLUS_ONE]] : f32
// CHECK:      %[[ABS:.*]] = math.absf %[[X]] : f32
// CHECK:      %[[USE_SMALL:.*]] = arith.cmpf olt, %[[ABS]]
// CHECK:      %[[OUT:.*]] = arith.select %[[USE_SMALL]], %{{.*}}, %[[LARGE]] : f32
// CHECK:      return %[[OUT]] : f32

// -----

// CHECK-LABEL: func.func @log1p_v2f16(
// CHECK-SAME:    %[[X:.*]]: vector<2xf16>
func.func @log1p_v2f16(%x: vector<2xf16>) -> vector<2xf16> {
  %r = math.log1p %x : vector<2xf16>
  return %r : vector<2xf16>
}
// CHECK-NOT:  math.log1p
// CHECK:      %[[ONE:.*]] = arith.constant dense<1.000000e+00> : vector<2xf16>
// CHECK:      %[[X_PLUS_ONE:.*]] = arith.addf %[[X]], %[[ONE]] : vector<2xf16>
// CHECK:      %[[LARGE:.*]] = math.log %[[X_PLUS_ONE]] : vector<2xf16>
// CHECK:      %[[ABS:.*]] = math.absf %[[X]] : vector<2xf16>
// CHECK:      %[[USE_SMALL:.*]] = arith.cmpf olt, %[[ABS]]
// CHECK:      %[[OUT:.*]] = arith.select %[[USE_SMALL]], %{{.*}}, %[[LARGE]] : vector<2xi1>, vector<2xf16>
// CHECK:      return %[[OUT]] : vector<2xf16>
