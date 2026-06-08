#include "objc_bridge.h"

void fail_marker_non_handle() {
  objc::retained<int> bad;
  (void)bad;
}
