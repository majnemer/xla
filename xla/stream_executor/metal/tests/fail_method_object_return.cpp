#include "objc_bridge.h"

void fail_method_object_return() {
  objc::method<objc::object<objc::NSString>()> bad("description");
  (void)bad;
}
