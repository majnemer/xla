#include "objc_bridge.h"

void fail_block_out_param() {
  objc::block<void(objc::out<objc::autoreleased<objc::NSString>>)> bad;
  (void)bad;
}
