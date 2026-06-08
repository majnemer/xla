#include "objc_bridge.h"

void fail_block_ownership_non_handle_return() {
  objc::block<objc::retained<int>()> bad;
  (void)bad;
}
