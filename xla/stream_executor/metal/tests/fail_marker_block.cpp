#include "objc_bridge.h"

void fail_marker_block() {
  objc::retained<objc::block<void()>> bad;
  (void)bad;
}
