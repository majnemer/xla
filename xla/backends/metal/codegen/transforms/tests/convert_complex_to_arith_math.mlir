// RUN: emitters_opt %s -xla-metal-convert-complex-to-arith-math -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @create_re_im(
// CHECK-SAME:    %[[R:.*]]: f32, %[[I:.*]]: f32
func.func @create_re_im(%r: f32, %i: f32) -> (f32, f32) {
  %z = complex.create %r, %i : complex<f32>
  %zr = complex.re %z : complex<f32>
  %zi = complex.im %z : complex<f32>
  return %zr, %zi : f32, f32
}
// CHECK: return %[[R]], %[[I]] : f32, f32

// -----

// CHECK-LABEL: func.func @complex_mul(
// CHECK-SAME:    %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<2xf32>
func.func @complex_mul(%a: complex<f32>, %b: complex<f32>) -> complex<f32> {
  %r = complex.mul %a, %b : complex<f32>
  return %r : complex<f32>
}
// CHECK-DAG: %[[AR:.*]] = vector.extract %[[A]][0] : f32 from vector<2xf32>
// CHECK-DAG: %[[AI:.*]] = vector.extract %[[A]][1] : f32 from vector<2xf32>
// CHECK-DAG: %[[BR:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK-DAG: %[[BI:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK-DAG: %[[RR:.*]] = arith.mulf %[[AR]], %[[BR]] : f32
// CHECK-DAG: %[[II:.*]] = arith.mulf %[[AI]], %[[BI]] : f32
// CHECK-DAG: %[[IR:.*]] = arith.mulf %[[AI]], %[[BR]] : f32
// CHECK-DAG: %[[RI:.*]] = arith.mulf %[[AR]], %[[BI]] : f32
// CHECK:     %[[REAL:.*]] = arith.subf %[[RR]], %[[II]] : f32
// CHECK:     %[[IMAG:.*]] = arith.addf %[[IR]], %[[RI]] : f32
// CHECK:     %[[OUT:.*]] = vector.from_elements %[[REAL]], %[[IMAG]] : vector<2xf32>
// CHECK:     return %[[OUT]] : vector<2xf32>

// -----

// CHECK-LABEL: func.func @complex_select(
// CHECK-SAME:    %[[P:.*]]: i1, %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<2xf32>
func.func @complex_select(%p: i1, %a: complex<f32>, %b: complex<f32>)
    -> complex<f32> {
  %r = arith.select %p, %a, %b : complex<f32>
  return %r : complex<f32>
}
// CHECK: %[[OUT:.*]] = arith.select %[[P]], %[[A]], %[[B]] : vector<2xf32>
// CHECK: return %[[OUT]] : vector<2xf32>

// -----

// CHECK-LABEL: func.func @tensor_extract(
// CHECK-SAME:    %[[ARG:.*]]: tensor<8xf32>, %[[I:.*]]: index
func.func @tensor_extract(%arg: tensor<4xcomplex<f32>>, %i: index)
    -> complex<f32> {
  %z = tensor.extract %arg[%i] : tensor<4xcomplex<f32>>
  return %z : complex<f32>
}
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[REAL_IDX:.*]] = arith.muli %[[I]], %[[TWO]] : index
// CHECK: %[[IMAG_IDX:.*]] = arith.addi %[[REAL_IDX]], %[[ONE]] : index
// CHECK: %[[REAL:.*]] = tensor.extract %[[ARG]]{{\[}}%[[REAL_IDX]]] : tensor<8xf32>
// CHECK: %[[IMAG:.*]] = tensor.extract %[[ARG]]{{\[}}%[[IMAG_IDX]]] : tensor<8xf32>
// CHECK: %[[OUT:.*]] = vector.from_elements %[[REAL]], %[[IMAG]] : vector<2xf32>
// CHECK: return %[[OUT]] : vector<2xf32>

// -----

// CHECK-LABEL: func.func @tensor_insert(
// CHECK-SAME:    %[[VAL:.*]]: vector<2xf32>, %[[DEST:.*]]: tensor<8xf32>, %[[I:.*]]: index
func.func @tensor_insert(%value: complex<f32>, %dest: tensor<4xcomplex<f32>>,
                         %i: index) -> tensor<4xcomplex<f32>> {
  %out = tensor.insert %value into %dest[%i] : tensor<4xcomplex<f32>>
  return %out : tensor<4xcomplex<f32>>
}
// CHECK-DAG: %[[REAL:.*]] = vector.extract %[[VAL]][0] : f32 from vector<2xf32>
// CHECK-DAG: %[[IMAG:.*]] = vector.extract %[[VAL]][1] : f32 from vector<2xf32>
// CHECK:     %[[WITH_REAL:.*]] = tensor.insert %[[REAL]] into %[[DEST]]
// CHECK:     %[[OUT:.*]] = tensor.insert %[[IMAG]] into %[[WITH_REAL]]
// CHECK:     return %[[OUT]] : tensor<8xf32>
