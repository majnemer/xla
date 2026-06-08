#include "objc_bridge.h"

void fail_block_facade_return() {
  objc::block<objc::NSString()> bad;
  (void)bad;
}
