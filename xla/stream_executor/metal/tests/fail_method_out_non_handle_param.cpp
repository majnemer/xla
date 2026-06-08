#include "objc_bridge.h"

void fail_method_out_non_handle_param() {
  objc::method<void(objc::out<objc::retained<int>>)> bad("setValue:");
  (void)bad;
}
