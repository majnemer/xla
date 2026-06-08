#include "objc_bridge.h"

void fail_send_ownership_marker(objc::NSObject receiver, SEL selector) {
  (void)objc::send<objc::retained<objc::NSString>>(receiver.ptr(), selector);
}
