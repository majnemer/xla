#include "objc_bridge.h"

void fail_block_out_return() {
  objc::block<objc::out<objc::retained<objc::NSString>>()> bad;
  (void)bad;
}
