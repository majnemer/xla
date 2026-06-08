#include <type_traits>

#include "objc_bridge.h"

namespace {

struct CommandBuffer : objc::NSObject {
  using objc::NSObject::NSObject;
};

struct Function : objc::NSObject {
  using objc::NSObject::NSObject;
};

[[maybe_unused]] void method_argument_checks(objc::NSObject receiver) {
  static_assert(std::is_same_v<
                decltype(std::declval<
                         objc::detail::method_arg_slot<short, int &&> &>()
                             .raw_arg()),
                short>);
  static_assert(std::is_same_v<
                decltype(std::declval<
                         objc::detail::method_arg_slot<objc::NSUInteger,
                                                        int &&> &>()
                             .raw_arg()),
                objc::NSUInteger>);
  static_assert(std::is_same_v<
                decltype(std::declval<
                         objc::detail::method_arg_slot<
                             objc::NSString,
                             objc::object<objc::NSString> &> &>()
                             .raw_arg()),
                void *>);

  objc::method<void(short)> takes_short("setShort:");
  takes_short(receiver, 1);

  objc::method<void(objc::NSUInteger)> takes_nsuinteger("setCount:");
  takes_nsuinteger(receiver, 1);

  objc::object<objc::NSString> name;
  objc::method<void(objc::NSString)> takes_string("setName:");
  takes_string(receiver, name);

  objc::block<void(CommandBuffer)> handler;
  objc::method<void(objc::block<void(CommandBuffer)>)> takes_block(
      "addCompletedHandler:");
  takes_block(receiver, handler);
  takes_block(receiver, nullptr);
}

[[maybe_unused]] void method_return_checks(objc::NSObject receiver) {
  objc::method<objc::retained<Function>(objc::NSString)> retained(
      "newFunctionWithName:");
  objc::object<Function> fn = retained(receiver, objc::NSString{});
  (void)fn;

  objc::method<objc::not_retained<objc::NSString>()> not_retained(
      "description");
  objc::NSString borrowed = not_retained.ref(receiver);
  objc::object<objc::NSString> owned = not_retained(receiver);
  (void)borrowed;
  (void)owned;

  objc::method<objc::autoreleased<objc::NSString>()> autoreleased(
      "localizedDescription");
  objc::object<objc::NSString> auto_owned = autoreleased(receiver);
  (void)auto_owned;

  objc::method<::id()> raw("description");
  ::id raw_result = raw(receiver);
  (void)raw_result;
}

[[maybe_unused]] void out_parameter_checks(objc::NSObject receiver) {
  objc::object<objc::NSError> error;
  objc::method<objc::retained<objc::NSString>(
      objc::out<objc::autoreleased<objc::NSError>>)>
      make_string("makeStringWithError:");
  objc::object<objc::NSString> result = make_string(receiver, error);
  (void)result;
}

[[maybe_unused]] void block_checks() {
  static_assert(std::is_same_v<
                decltype(std::declval<
                         const objc::block<objc::retained<objc::NSString>()> &>()()),
                objc::object<objc::NSString>>);
  static_assert(std::is_same_v<
                decltype(std::declval<
                         const objc::block<
                             objc::not_retained<objc::NSString>()> &>()()),
                objc::object<objc::NSString>>);
  static_assert(std::is_same_v<
                decltype(std::declval<
                         const objc::block<
                             objc::autoreleased<objc::NSString>()> &>()()),
                objc::object<objc::NSString>>);

  auto retained = objc::make_block<objc::retained<objc::NSString>()>(
      []() -> objc::object<objc::NSString> { return {}; });

  objc::object<objc::NSString> cached;
  auto not_retained = objc::make_block<objc::not_retained<objc::NSString>()>(
      [cached]() -> objc::NSString { return cached; });

  auto autoreleased = objc::make_block<objc::autoreleased<objc::NSString>()>(
      []() -> objc::object<objc::NSString> { return {}; });

  auto facade_param = objc::make_block<void(objc::NSString)>(
      [](objc::NSString value) { (void)value; });

  auto scalar = objc::make_block<int(short)>(
      [](short value) { return static_cast<int>(value); });

  (void)retained;
  (void)not_retained;
  (void)autoreleased;
  (void)facade_param;
  (void)scalar;
}

}  // namespace
