// RUN: emitters_opt %s -xla-metal-lower-float-storage -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @store_bf16(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi16>
// CHECK-SAME:    %[[VALUE:.*]]: f32
// CHECK-SAME:  ) -> tensor<4xi16>
func.func @store_bf16(%arg0: tensor<4xbf16>, %value: f32) -> tensor<4xbf16> {
  %idx = arith.constant 0 : index
  %bf16 = arith.truncf %value : f32 to bf16
  %out = tensor.insert %bf16 into %arg0[%idx] : tensor<4xbf16>
  return %out : tensor<4xbf16>
}
// CHECK:      %[[BITS:.*]] = arith.bitcast %[[VALUE]]
// CHECK:      %[[HIGH:.*]] = arith.shrui %[[BITS]]
// CHECK:      %[[ABS:.*]] = arith.andi %[[BITS]]
// CHECK:      %[[IS_NAN:.*]] = arith.cmpi ugt, %[[ABS]]
// CHECK:      %[[QUIET_NAN:.*]] = arith.ori %[[HIGH]]
// CHECK:      %[[ROUNDED:.*]] = arith.shrui
// CHECK:      %[[SELECT:.*]] = arith.select %[[IS_NAN]], %[[QUIET_NAN]], %[[ROUNDED]]
// CHECK:      %[[TRUNC:.*]] = arith.trunci %[[SELECT]] : i32 to i16
// CHECK:      %[[OUT:.*]] = tensor.insert %[[TRUNC]] into %[[ARG]]
// CHECK:      return %[[OUT]] : tensor<4xi16>

// CHECK-LABEL: func.func @load_bf16(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi16>
// CHECK-SAME:  ) -> f32
func.func @load_bf16(%arg0: tensor<4xbf16>) -> f32 {
  %idx = arith.constant 0 : index
  %bf16 = tensor.extract %arg0[%idx] : tensor<4xbf16>
  %value = arith.extf %bf16 : bf16 to f32
  return %value : f32
}
// CHECK:      %[[BITS16:.*]] = tensor.extract %[[ARG]]
// CHECK:      %[[BITS32:.*]] = arith.extui %[[BITS16]] : i16 to i32
// CHECK:      %[[SHIFTED:.*]] = arith.shli %[[BITS32]]
// CHECK:      %[[VALUE:.*]] = arith.bitcast %[[SHIFTED]] : i32 to f32
// CHECK:      return %[[VALUE]] : f32

// CHECK-LABEL: func.func @constant_bf16() -> i16
func.func @constant_bf16() -> bf16 {
  %c = arith.constant 1.500000e+00 : bf16
  return %c : bf16
}
// CHECK:      %[[C:.*]] = arith.constant
// CHECK-SAME:   : i16
// CHECK:      return %[[C]] : i16

// CHECK-LABEL: func.func @constant_vector_bf16() -> vector<2xi16>
func.func @constant_vector_bf16() -> vector<2xbf16> {
  %c = arith.constant dense<[0.000000e+00, 1.000000e+00]> : vector<2xbf16>
  return %c : vector<2xbf16>
}
// CHECK:      %[[C:.*]] = arith.constant
// CHECK-SAME:   : vector<2xi16>
// CHECK:      return %[[C]] : vector<2xi16>

// CHECK-LABEL: func.func @vector_from_elements(
// CHECK-SAME:    %[[A:.*]]: f32, %[[B:.*]]: f32
// CHECK-SAME:  ) -> vector<2xi16>
func.func @vector_from_elements(%a: f32, %b: f32) -> vector<2xbf16> {
  %a_bf16 = arith.truncf %a : f32 to bf16
  %b_bf16 = arith.truncf %b : f32 to bf16
  %out = vector.from_elements %a_bf16, %b_bf16 : vector<2xbf16>
  return %out : vector<2xbf16>
}
// CHECK:      %[[A_BITS:.*]] = arith.bitcast %[[A]]
// CHECK:      %[[A_BF16:.*]] = arith.trunci {{.*}} : i32 to i16
// CHECK:      %[[B_BITS:.*]] = arith.bitcast %[[B]]
// CHECK:      %[[B_BF16:.*]] = arith.trunci {{.*}} : i32 to i16
// CHECK:      %[[OUT:.*]] = vector.from_elements %[[A_BF16]], %[[B_BF16]] : vector<2xi16>
// CHECK:      return %[[OUT]] : vector<2xi16>

// CHECK-LABEL: func.func @vector_extract_bf16(
// CHECK-SAME:    %[[VEC:.*]]: vector<4xi16>
// CHECK-SAME:  ) -> i16
func.func @vector_extract_bf16(%vec: vector<4xbf16>) -> bf16 {
  %out = vector.extract %vec[0] : bf16 from vector<4xbf16>
  return %out : bf16
}
// CHECK:      %[[OUT:.*]] = vector.extract %[[VEC]][0] : i16 from vector<4xi16>
// CHECK:      return %[[OUT]] : i16

// CHECK-LABEL: func.func @vector_insert_bf16(
// CHECK-SAME:    %[[VEC:.*]]: vector<4xi16>, %[[VALUE:.*]]: i16
// CHECK-SAME:  ) -> vector<4xi16>
func.func @vector_insert_bf16(%vec: vector<4xbf16>, %value: bf16) -> vector<4xbf16> {
  %out = vector.insert %value, %vec[1] : bf16 into vector<4xbf16>
  return %out : vector<4xbf16>
}
// CHECK:      %[[OUT:.*]] = vector.insert %[[VALUE]], %[[VEC]] [1] : i16 into vector<4xi16>
// CHECK:      return %[[OUT]] : vector<4xi16>

// CHECK-LABEL: func.func @vector_broadcast_bf16(
// CHECK-SAME:    %[[VALUE:.*]]: i16
// CHECK-SAME:  ) -> vector<4xi16>
func.func @vector_broadcast_bf16(%value: bf16) -> vector<4xbf16> {
  %out = vector.broadcast %value : bf16 to vector<4xbf16>
  return %out : vector<4xbf16>
}
// CHECK:      %[[OUT:.*]] = vector.broadcast %[[VALUE]] : i16 to vector<4xi16>
// CHECK:      return %[[OUT]] : vector<4xi16>

// CHECK-LABEL: func.func @select_bf16(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: i16, %[[B:.*]]: i16
// CHECK-SAME:  ) -> i16
func.func @select_bf16(%cond: i1, %a: bf16, %b: bf16) -> bf16 {
  %out = arith.select %cond, %a, %b : bf16
  return %out : bf16
}
// CHECK:      %[[OUT:.*]] = arith.select %[[COND]], %[[A]], %[[B]] : i16
// CHECK:      return %[[OUT]] : i16

// CHECK-LABEL: func.func @vector_transfer_read(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi16>
// CHECK-SAME:    %[[PAD_VALUE:.*]]: f32
// CHECK-SAME:  ) -> vector<4xi16>
func.func @vector_transfer_read(%arg0: tensor<4xbf16>, %pad_value: f32) -> vector<4xbf16> {
  %idx = arith.constant 0 : index
  %pad = arith.truncf %pad_value : f32 to bf16
  %out = vector.transfer_read %arg0[%idx], %pad {in_bounds = [true]} : tensor<4xbf16>, vector<4xbf16>
  return %out : vector<4xbf16>
}
// CHECK:      %[[PAD_BITS:.*]] = arith.bitcast %[[PAD_VALUE]]
// CHECK:      %[[PAD:.*]] = arith.trunci {{.*}} : i32 to i16
// CHECK:      %[[READ:.*]] = vector.transfer_read %[[ARG]]{{\[.*\]}}, %[[PAD]] {in_bounds = [true]} : tensor<4xi16>, vector<4xi16>
// CHECK:      return %[[READ]] : vector<4xi16>

// CHECK-LABEL: func.func @vector_transfer_write(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi16>
// CHECK-SAME:    %[[A:[^,]+]]: f32
// CHECK-SAME:  ) -> tensor<4xi16>
func.func @vector_transfer_write(%arg0: tensor<4xbf16>, %a: f32, %b: f32, %c: f32, %d: f32) -> tensor<4xbf16> {
  %idx = arith.constant 0 : index
  %a_bf16 = arith.truncf %a : f32 to bf16
  %b_bf16 = arith.truncf %b : f32 to bf16
  %c_bf16 = arith.truncf %c : f32 to bf16
  %d_bf16 = arith.truncf %d : f32 to bf16
  %vec = vector.from_elements %a_bf16, %b_bf16, %c_bf16, %d_bf16 : vector<4xbf16>
  %out = vector.transfer_write %vec, %arg0[%idx] {in_bounds = [true]} : vector<4xbf16>, tensor<4xbf16>
  return %out : tensor<4xbf16>
}
// CHECK:      %[[A_BITS:.*]] = arith.bitcast %[[A]]
// CHECK:      %[[A_BF16:.*]] = arith.trunci {{.*}} : i32 to i16
// CHECK:      %[[VEC:.*]] = vector.from_elements %[[A_BF16]]
// CHECK-SAME:   : vector<4xi16>
// CHECK:      %[[OUT:.*]] = vector.transfer_write %[[VEC]], %[[ARG]]{{\[.*\]}} {in_bounds = [true]} : vector<4xi16>, tensor<4xi16>
// CHECK:      return %[[OUT]] : tensor<4xi16>

// CHECK-LABEL: func.func @store_f8e5m2(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi8>
// CHECK-SAME:    %[[BITS:.*]]: i8
// CHECK-SAME:  ) -> tensor<4xi8>
func.func @store_f8e5m2(%arg0: tensor<4xf8E5M2>, %bits: i8) -> tensor<4xf8E5M2> {
  %idx = arith.constant 0 : index
  %f8 = arith.bitcast %bits : i8 to f8E5M2
  %out = tensor.insert %f8 into %arg0[%idx] : tensor<4xf8E5M2>
  return %out : tensor<4xf8E5M2>
}
// CHECK:      %[[OUT:.*]] = tensor.insert %[[BITS]] into %[[ARG]]
// CHECK:      return %[[OUT]] : tensor<4xi8>

// CHECK-LABEL: func.func @constant_f8e5m2() -> i8
func.func @constant_f8e5m2() -> f8E5M2 {
  %c = arith.constant 1.000000e+00 : f8E5M2
  return %c : f8E5M2
}
// CHECK:      %[[C:.*]] = arith.constant
// CHECK-SAME:   : i8
// CHECK:      return %[[C]] : i8

// CHECK-LABEL: func.func @vector_f8e5m2(
// CHECK-SAME:    %[[A:[^,]+]]: i8, %[[B:.*]]: i8
// CHECK-SAME:  ) -> vector<2xi8>
func.func @vector_f8e5m2(%a: i8, %b: i8) -> vector<2xf8E5M2> {
  %a_f8 = arith.bitcast %a : i8 to f8E5M2
  %b_f8 = arith.bitcast %b : i8 to f8E5M2
  %out = vector.from_elements %a_f8, %b_f8 : vector<2xf8E5M2>
  return %out : vector<2xf8E5M2>
}
// CHECK:      %[[OUT:.*]] = vector.from_elements %[[A]], %[[B]] : vector<2xi8>
// CHECK:      return %[[OUT]] : vector<2xi8>

// CHECK-LABEL: func.func @poison_bf16() -> i16
func.func @poison_bf16() -> bf16 {
  %p = ub.poison : bf16
  return %p : bf16
}
// CHECK:      %[[P:.*]] = ub.poison : i16
// CHECK:      return %[[P]] : i16
