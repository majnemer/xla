#include "objc_bridge.h"

void fail_block_not_retained_owner_return() {
  auto bad = objc::make_block<objc::not_retained<objc::NSString>()>(
      []() -> objc::object<objc::NSString> { return {}; });
  (void)bad;
}
