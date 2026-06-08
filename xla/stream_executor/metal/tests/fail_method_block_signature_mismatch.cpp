#include "objc_bridge.h"

namespace {

struct CommandBuffer : objc::NSObject {
  using objc::NSObject::NSObject;
};

}  // namespace

void fail_method_block_signature_mismatch(objc::NSObject receiver) {
  objc::method<void(objc::block<void(CommandBuffer)>)> add_completed_handler(
      "addCompletedHandler:");
  objc::block<void()> wrong_handler;
  add_completed_handler(receiver, wrong_handler);
}
