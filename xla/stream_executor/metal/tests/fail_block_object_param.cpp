#include "objc_bridge.h"

void fail_block_object_param() {
  objc::block<void(objc::object<objc::NSString>)> bad;
  (void)bad;
}
