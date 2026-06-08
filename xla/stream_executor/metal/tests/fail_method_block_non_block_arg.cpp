#include "objc_bridge.h"

void fail_method_block_non_block_arg(objc::NSObject receiver) {
  objc::method<void(objc::block<void()>)> set_handler("setHandler:");
  objc::NSString not_a_block;
  set_handler(receiver, not_a_block);
}
