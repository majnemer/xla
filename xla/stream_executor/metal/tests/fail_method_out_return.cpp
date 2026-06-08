#include "objc_bridge.h"

void fail_method_out_return() {
  objc::method<objc::out<objc::retained<objc::NSString>>()> bad("value");
  (void)bad;
}
