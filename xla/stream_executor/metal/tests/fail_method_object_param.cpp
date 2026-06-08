#include "objc_bridge.h"

void fail_method_object_param() {
  objc::method<void(objc::object<objc::NSString>)> bad("setName:");
  (void)bad;
}
