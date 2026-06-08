#include "objc_bridge.h"

void fail_send_out_marker(objc::NSObject receiver, SEL selector) {
  (void)objc::send<objc::out<objc::retained<objc::NSString>>>(
      receiver.ptr(), selector);
}
