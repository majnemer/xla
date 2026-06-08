#include "objc_bridge.h"

void fail_method_ownership_non_handle_return() {
  objc::method<objc::retained<int>()> bad("value");
  (void)bad;
}
