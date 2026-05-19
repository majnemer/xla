// RUN: emitters_opt %s -xla-metal-lower-sub-byte-storage -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_u4_to_u8(
// CHECK-SAME:    %[[SRC:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[DST:.*]]: tensor<4xi8>
// CHECK-SAME:  ) -> tensor<4xi8>
func.func @convert_u4_to_u8(%src: tensor<4xi4>, %dst: tensor<4xi8>) -> tensor<4xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i4
  %v = vector.transfer_read %src[%c0], %pad {in_bounds = [true]} : tensor<4xi4>, vector<4xi4>
  %lane = vector.extract %v[0] : i4 from vector<4xi4>
  %wide = arith.extui %lane : i4 to i8
  %out = tensor.insert %wide into %dst[%c0] : tensor<4xi8>
  return %out : tensor<4xi8>
}
// CHECK:      %[[PACKED:.*]] = tensor.extract %[[SRC]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      %[[WIDE:.*]] = arith.andi %[[PACKED]]
// CHECK:      %[[OUT:.*]] = tensor.insert %[[WIDE]] into %[[DST]]
// CHECK-SAME:   : tensor<4xi8>
// CHECK:      return %[[OUT]] : tensor<4xi8>

// -----

// CHECK-LABEL: func.func @load_vector_i4(
// CHECK-SAME:    %[[SRC:.*]]: tensor<5xi8>
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:  ) -> vector<4xi8>
func.func @load_vector_i4(%src: tensor<9xi4>, %i: index) -> vector<4xi4> {
  %pad = arith.constant 0 : i4
  %v = vector.transfer_read %src[%i], %pad {in_bounds = [true]} : tensor<9xi4>, vector<4xi4>
  return %v : vector<4xi4>
}
// CHECK:      tensor.extract %[[SRC]]
// CHECK-SAME:   : tensor<5xi8>
// CHECK:      arith.shrui
// CHECK:      arith.andi
// CHECK:      vector.from_elements
// CHECK-SAME:   : vector<4xi8>

// -----

// CHECK-LABEL: func.func @store_i4(
// CHECK-SAME:    %[[DST:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:  ) -> tensor<2xi8>
func.func @store_i4(%dst: tensor<4xi4>, %value: i8) -> tensor<4xi4> {
  %c1 = arith.constant 1 : index
  %narrow = arith.trunci %value : i8 to i4
  %out = tensor.insert %narrow into %dst[%c1] : tensor<4xi4>
  return %out : tensor<4xi4>
}
// CHECK:      %[[LOW:.*]] = arith.andi %[[VALUE]], %{{.*}} : i8
// CHECK:      %[[SHIFTED_VALUE:.*]] = arith.shli %[[LOW]]
// CHECK:      %[[OUT:.*]] = xla.atomic_rmw %[[DST]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      ^bb0(%[[CURRENT:.*]]: i8):
// CHECK:        %[[PRESERVED:.*]] = arith.andi %[[CURRENT]]
// CHECK:        %[[MERGED:.*]] = arith.ori %[[PRESERVED]], %[[SHIFTED_VALUE]]
// CHECK:        xla.yield %[[MERGED]] : i8
// CHECK:      return %[[OUT]] : tensor<2xi8>

// -----

// CHECK-LABEL: func.func @store_vector_i4(
// CHECK-SAME:    %[[DST:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[VALUE:.*]]: vector<2xi8>
// CHECK-SAME:  ) -> tensor<2xi8>
func.func @store_vector_i4(%dst: tensor<4xi4>, %value: vector<2xi4>) -> tensor<4xi4> {
  %c0 = arith.constant 0 : index
  %out = vector.transfer_write %value, %dst[%c0] {in_bounds = [true]} : vector<2xi4>, tensor<4xi4>
  return %out : tensor<4xi4>
}
// CHECK:      vector.extract %[[VALUE]][0] : i8 from vector<2xi8>
// CHECK:      xla.atomic_rmw %[[DST]]
// CHECK:      vector.extract %[[VALUE]][1] : i8 from vector<2xi8>
// CHECK:      xla.atomic_rmw

// -----

// CHECK-LABEL: func.func @constant_i4() -> tensor<2xi8>
func.func @constant_i4() -> tensor<3xi4> {
  %c = arith.constant dense<[1, 2, 3]> : tensor<3xi4>
  return %c : tensor<3xi4>
}
// CHECK:      %[[C:.*]] = arith.constant dense<[33, 3]> : tensor<2xi8>
// CHECK:      return %[[C]] : tensor<2xi8>

// -----

// CHECK-LABEL: func.func @f4_extract_bits(
// CHECK-SAME:    %[[F4SRC:.*]]: tensor<2xi8>
func.func @f4_extract_bits(%src: tensor<3xf4E2M1FN>, %i: index) -> i8 {
  %x = tensor.extract %src[%i] : tensor<3xf4E2M1FN>
  %bits = arith.bitcast %x : f4E2M1FN to i4
  %wide = arith.extui %bits : i4 to i8
  return %wide : i8
}
// CHECK:      %[[F4PACKED:.*]] = tensor.extract %[[F4SRC]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      %[[F4SHIFTED:.*]] = arith.shrui %[[F4PACKED]]
// CHECK:      %[[F4LOW:.*]] = arith.andi %[[F4SHIFTED]]
// CHECK:      return %[[F4LOW]] : i8
