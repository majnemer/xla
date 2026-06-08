#include "objc_bridge.h"

void fail_method_out_bad_param() {
  objc::method<void(objc::out<int>)> bad("setValue:");
  (void)bad;
}
