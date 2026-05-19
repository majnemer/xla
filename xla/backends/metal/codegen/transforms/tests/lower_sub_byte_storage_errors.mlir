// RUN: not emitters_opt %s -xla-metal-lower-sub-byte-storage 2>&1 | FileCheck %s

func.func @transfer_read_out_of_bounds(%src: tensor<9xi4>, %i: index) -> vector<4xi4> {
  %pad = arith.constant 0 : i4
  %v = vector.transfer_read %src[%i], %pad {in_bounds = [false]} : tensor<9xi4>, vector<4xi4>
  return %v : vector<4xi4>
}

// CHECK: failed to legalize operation 'vector.transfer_read'
