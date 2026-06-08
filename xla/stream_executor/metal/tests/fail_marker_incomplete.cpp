#include "objc_bridge.h"

struct Forward;

void fail_marker_incomplete() {
  objc::retained<Forward> bad;
  (void)bad;
}
