#include "objc_bridge.h"

void fail_method_facade_return() {
  objc::method<objc::NSString()> bad("description");
  (void)bad;
}
