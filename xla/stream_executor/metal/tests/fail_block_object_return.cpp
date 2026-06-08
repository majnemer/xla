#include "objc_bridge.h"

void fail_block_object_return() {
  objc::block<objc::object<objc::NSString>()> bad;
  (void)bad;
}
