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

// CHECK-LABEL: func.func @atomic_rmw_i4(
// CHECK-SAME:    %[[DST:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:  ) -> tensor<2xi8>
func.func @atomic_rmw_i4(%dst: tensor<4xi4>, %i: index, %value: i4) -> tensor<4xi4> {
  %out = xla.atomic_rmw %dst[%i] : tensor<4xi4> {
    ^bb0(%current: i4):
      %sum = arith.addi %current, %value : i4
      xla.yield %sum : i4
  }
  return %out : tensor<4xi4>
}
// CHECK:      %[[OUT:.*]] = xla.atomic_rmw %[[DST]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      ^bb0(%[[CURRENT:.*]]: i8):
// CHECK-NOT:    i4
// CHECK:        %[[SHIFTED:.*]] = arith.shrui %[[CURRENT]]
// CHECK:        %[[LOW_BITS:.*]] = arith.andi %[[SHIFTED]]
// CHECK:        %[[VALUE_LOW:.*]] = arith.andi %[[VALUE]]
// CHECK:        %[[SUM:.*]] = arith.addi %[[LOW_BITS]], %[[VALUE_LOW]] : i8
// CHECK:        %[[SUM_LOW_BITS:.*]] = arith.andi %[[SUM]]
// CHECK:        arith.shli %[[SUM_LOW_BITS]]
// CHECK-NOT:    i4
// CHECK:        xla.yield {{.*}} : i8
// CHECK:      return %[[OUT]] : tensor<2xi8>

// -----

// CHECK-LABEL: func.func @atomic_rmw_i4_nested_capture(
// CHECK-SAME:    %[[DST:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:    %[[PRED:.*]]: i1
// CHECK-SAME:  ) -> tensor<2xi8>
func.func @atomic_rmw_i4_nested_capture(%dst: tensor<4xi4>, %i: index, %value: i4, %pred: i1) -> tensor<4xi4> {
  %out = xla.atomic_rmw %dst[%i] : tensor<4xi4> {
    ^bb0(%current: i4):
      %selected = scf.if %pred -> (i4) {
        %sum = arith.addi %current, %value : i4
        scf.yield %sum : i4
      } else {
        scf.yield %value : i4
      }
      xla.yield %selected : i4
  }
  return %out : tensor<4xi4>
}
// CHECK:      %[[OUT:.*]] = xla.atomic_rmw %[[DST]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      ^bb0(%[[CURRENT:.*]]: i8):
// CHECK-NOT:    i4
// CHECK:        %[[SHIFTED:.*]] = arith.shrui %[[CURRENT]]
// CHECK:        %[[LOW_BITS:.*]] = arith.andi %[[SHIFTED]]
// CHECK:        %[[VALUE_LOW:.*]] = arith.andi %[[VALUE]]
// CHECK:        %[[SELECTED:.*]] = scf.if %[[PRED]] -> (i8) {
// CHECK:          %[[SUM:.*]] = arith.addi %[[LOW_BITS]], %[[VALUE_LOW]] : i8
// CHECK:          %[[SUM_LOW:.*]] = arith.andi %[[SUM]]
// CHECK:          scf.yield %[[SUM_LOW]] : i8
// CHECK:        } else {
// CHECK:          scf.yield %[[VALUE_LOW]] : i8
// CHECK:        }
// CHECK-NOT:    i4
// CHECK:        xla.yield {{.*}} : i8
// CHECK:      return %[[OUT]] : tensor<2xi8>

// -----

// CHECK-LABEL: func.func @atomic_rmw_i4_signed_cmp(
// CHECK-SAME:    %[[DST:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:  ) -> tensor<2xi8>
func.func @atomic_rmw_i4_signed_cmp(%dst: tensor<4xi4>, %i: index, %value: i4) -> tensor<4xi4> {
  %out = xla.atomic_rmw %dst[%i] : tensor<4xi4> {
    ^bb0(%current: i4):
      %zero = arith.constant 0 : i4
      %negative = arith.cmpi slt, %current, %zero : i4
      %selected = arith.select %negative, %value, %current : i4
      xla.yield %selected : i4
  }
  return %out : tensor<4xi4>
}
// CHECK:      %[[OUT:.*]] = xla.atomic_rmw %[[DST]]
// CHECK-SAME:   : tensor<2xi8>
// CHECK:      ^bb0(%[[CURRENT:.*]]: i8):
// CHECK-NOT:    i4
// CHECK:        %[[SHIFTED:.*]] = arith.shrui %[[CURRENT]]
// CHECK:        %[[LOW_BITS:.*]] = arith.andi %[[SHIFTED]]
// CHECK:        %[[SIGN_FLIPPED:.*]] = arith.xori %[[LOW_BITS]]
// CHECK:        %[[SIGNED_CURRENT:.*]] = arith.subi %[[SIGN_FLIPPED]]
// CHECK:        %[[NEGATIVE:.*]] = arith.cmpi slt, %[[SIGNED_CURRENT]]
// CHECK:        arith.select %[[NEGATIVE]]
// CHECK-NOT:    i4
// CHECK:        xla.yield {{.*}} : i8
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

// -----

// CHECK-LABEL: func.func @callee_i4(
// CHECK-SAME:    %[[ARG:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:  ) -> i8
func.func @callee_i4(%arg0: tensor<4xi4>, %value: i4) -> i4 {
  return %value : i4
}
// CHECK:      return %[[VALUE]] : i8

// CHECK-LABEL: func.func @caller_i4(
// CHECK-SAME:    %[[ARG:.*]]: tensor<2xi8>
// CHECK-SAME:    %[[VALUE:.*]]: i8
// CHECK-SAME:  ) -> i8
func.func @caller_i4(%arg0: tensor<4xi4>, %value: i4) -> i4 {
  %out = func.call @callee_i4(%arg0, %value) : (tensor<4xi4>, i4) -> i4
  return %out : i4
}
// CHECK:      %[[OUT:.*]] = call @callee_i4(%[[ARG]], %[[VALUE]]) : (tensor<2xi8>, i8) -> i8
// CHECK:      return %[[OUT]] : i8
