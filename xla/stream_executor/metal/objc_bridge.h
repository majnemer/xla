// objc_bridge.hpp — C++17 pybind-flavored Objective-C runtime bridge.
//
// Version: 0.7.3
//
// Goal
// ----
// Make Objective-C objects usable from plain C++ without Objective-C syntax,
// with explicit ownership at every boundary.
//
//   objc::handle              non-owning, dynamically typed ObjC object handle
//   Objective-C facades       non-owning, statically typed handle values
//   objc::object<T>           owning retained ObjC object wrapper value
//   objc::retain<T>(p)        p is +0/not-retained; retain it
//   objc::adopt<T>(p)         p is +1/retained; adopt it
//   objc::ref<T>(p)           p is +0/not-retained; borrowed facade
//
//   objc::retained<T>         method returns T object at +1; adopt it
//   objc::not_retained<T>     method returns T object at +0; retain it
//   objc::autoreleased<T>     method returns T object at +0 autoreleased; retain it
//
//   objc::method<Sig>         typed, cached selector with ownership-aware
//   return objc::selector<Sig>       typed, cached selector, raw send-style
//   objc::protocol<P>         typed wrapper over id<P>-style protocol objects
//
//   ptr objc::alloc_init<T>(...)  alloc + init as one ownership transfer
//
// C++ standard: C++17.
// Targets: Apple Objective-C runtime on arm64, x86_64, i386, armv7.
//
// Design notes
// ------------
// This follows the pybind11 value-wrapper model: Objective-C heap objects are
// not treated as C++ objects. A wrapper value contains an Objective-C pointer.
// Facades such as NSString are borrowed/non-owning handle values over an
// Objective-C pointer and inherit directly from `handle`. `object<T>` is the
// ownership-bearing form. It uses composition, not inheritance, and stores a T
// facade subobject, so temporary owners cannot silently slice into dangling
// borrowed facades while lvalue owners still support `owner->method()`.
//
// This header intentionally does not require Objective-C syntax. It uses the
// Objective-C runtime C API and typed objc_msgSend casts.

#pragma once

#define OBJC_BRIDGE_VERSION "0.7.3"
#define OBJC_BRIDGE_VERSION_MAJOR 0
#define OBJC_BRIDGE_VERSION_MINOR 7
#define OBJC_BRIDGE_VERSION_PATCH 3

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

#if defined(__clang__) && __has_cpp_attribute(clang::lifetimebound)
#define OBJC_LIFETIMEBOUND [[clang::lifetimebound]]
#else
#define OBJC_LIFETIMEBOUND
#endif

#if __has_cpp_attribute(gsl::Pointer)
#define OBJC_GSL_POINTER [[gsl::Pointer]]
#else
#define OBJC_GSL_POINTER
#endif

#if __has_cpp_attribute(gsl::Owner)
#define OBJC_GSL_OWNER [[gsl::Owner]]
#else
#define OBJC_GSL_OWNER
#endif

#include <cstddef>
#include <cstdlib>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Block.h>
#include <objc/message.h>
#include <objc/runtime.h>

namespace objc {

#if __LP64__
using NSInteger = long;
using NSUInteger = unsigned long;
#else
using NSInteger = int;
using NSUInteger = unsigned int;
#endif

using const_id = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<::id>>>;

// ===========================================================================
// ABI traits
// ===========================================================================

template <typename T>
struct indirect_return;

namespace detail {

template <typename>
inline constexpr bool always_false_v = false;

template <typename T>
using bare_t = std::remove_cv_t<T>;

template <typename T>
constexpr bool default_indirect_return() {
  if constexpr (std::is_void_v<T> || std::is_scalar_v<T>) {
    return false;
  } else {
#if __has_builtin(__can_pass_in_regs)
    if constexpr (std::is_class_v<T> || std::is_union_v<T>) {
      if (!__can_pass_in_regs(T)) {
        return true;
      }
    }
#endif

#if defined(__x86_64__)
    return sizeof(T) > 16 || alignof(T) > 16;
#elif defined(__i386__)
    return !(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
#elif defined(__arm64__) || defined(__aarch64__)
    return sizeof(T) > 16 || alignof(T) > 16;
#elif defined(__arm__)
    return sizeof(T) > 4 || alignof(T) > 4;
#else
#error "Unsupported architecture"
#endif
  }
}

template <typename T>
constexpr bool uses_fpret() {
#if defined(__x86_64__)
  return std::is_same_v<bare_t<T>, long double>;
#elif defined(__i386__)
  return std::is_same_v<bare_t<T>, float> || std::is_same_v<bare_t<T>, double> ||
         std::is_same_v<bare_t<T>, long double>;
#else
  return false;
#endif
}

template <typename T>
constexpr bool uses_fp2ret() {
#if defined(__x86_64__)
  return std::is_same_v<bare_t<T>, __complex__ long double>;
#else
  return false;
#endif
}

}  // namespace detail

template <typename T>
struct indirect_return : std::bool_constant<detail::default_indirect_return<T>()> {};

template <typename T>
inline constexpr bool indirect_return_v = indirect_return<T>::value;

template <typename T>
inline constexpr bool uses_fpret_v = detail::uses_fpret<T>();

template <typename T>
inline constexpr bool uses_fp2ret_v = detail::uses_fp2ret<T>();

// ===========================================================================
// Selector / class / protocol caching
// ===========================================================================

#define OBJC_SEL(name)                     \
  ([]() -> SEL {                           \
    static SEL s = sel_registerName(name); \
    return s;                              \
  }())

#define OBJC_CLASS(name)                  \
  ([]() -> Class {                        \
    static Class c = objc_getClass(name); \
    return c;                             \
  }())

#define OBJC_PROTOCOL(name)                      \
  ([]() -> Protocol * {                          \
    static Protocol *p = objc_getProtocol(name); \
    return p;                                    \
  }())

// ===========================================================================
// Forward declarations
// ===========================================================================

class handle;

template <class T>
class object;

template <class Signature>
class block;

template <class T>
struct retained;

template <class T>
struct not_retained;

template <class T>
struct autoreleased;

template <class OwnershipReturn>
struct out;

// ===========================================================================
// Message dispatch
// ===========================================================================

namespace detail {

template <typename Ret, typename... Args>
inline Ret call_msgSend(const void *self, SEL op, Args... args) {
  using Fn = Ret (*)(const void *, SEL, Args...);
  return reinterpret_cast<Fn>(&objc_msgSend)(self, op, args...);
}

template <typename Ret, typename... Args>
inline Ret call_fpret(const void *self, SEL op, Args... args) {
#if defined(__x86_64__) || defined(__i386__)
  using Fn = Ret (*)(const void *, SEL, Args...);
  return reinterpret_cast<Fn>(&objc_msgSend_fpret)(self, op, args...);
#else
  (void)self;
  (void)op;
  ((void)args, ...);
  static_assert(always_false_v<Ret>, "objc_msgSend_fpret only exists on i386/x86_64");
#endif
}

template <typename Ret, typename... Args>
inline Ret call_fp2ret(const void *self, SEL op, Args... args) {
#if defined(__x86_64__)
  using Fn = Ret (*)(const void *, SEL, Args...);
  return reinterpret_cast<Fn>(&objc_msgSend_fp2ret)(self, op, args...);
#else
  (void)self;
  (void)op;
  ((void)args, ...);
  static_assert(always_false_v<Ret>, "objc_msgSend_fp2ret only exists on x86_64");
#endif
}

template <typename Ret, typename... Args>
inline Ret call_indirect(const void *self, SEL op, Args... args) {
#if defined(__x86_64__) || defined(__i386__) || defined(__arm__)
  Ret r;
  using Fn = void (*)(Ret *, const void *, SEL, Args...);
  reinterpret_cast<Fn>(&objc_msgSend_stret)(&r, self, op, args...);
  return r;
#elif defined(__arm64__) || defined(__aarch64__)
  return call_msgSend<Ret>(self, op, args...);
#else
#error "Unsupported architecture"
#endif
}

template <class T, class = void>
struct has_objc_arg : std::false_type {};

template <class T>
struct has_objc_arg<T, std::void_t<decltype(std::declval<const T &>().objc_arg())>>
    : std::true_type {};

template <class T>
inline constexpr bool has_objc_arg_v =
    has_objc_arg<std::remove_cv_t<std::remove_reference_t<T>>>::value;

template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T>
struct is_object_wrapper : std::false_type {};

template <class T>
struct is_object_wrapper<object<T>> : std::true_type {
  using element_type = T;
};

template <class T>
inline constexpr bool is_object_wrapper_v = is_object_wrapper<remove_cvref_t<T>>::value;

template <class T>
struct is_block_wrapper : std::false_type {};

template <class Signature>
struct is_block_wrapper<block<Signature>> : std::true_type {
  using signature_type = Signature;
};

template <class T>
inline constexpr bool is_block_wrapper_v = is_block_wrapper<remove_cvref_t<T>>::value;

template <class T>
struct is_objc_ownership_return : std::false_type {};

template <class T>
struct is_objc_ownership_return<retained<T>> : std::bool_constant<(sizeof(retained<T>) > 0)> {};

template <class T>
struct is_objc_ownership_return<not_retained<T>>
    : std::bool_constant<(sizeof(not_retained<T>) > 0)> {};

template <class T>
struct is_objc_ownership_return<autoreleased<T>>
    : std::bool_constant<(sizeof(autoreleased<T>) > 0)> {};

template <class T>
inline constexpr bool is_objc_ownership_return_v =
    is_objc_ownership_return<remove_cvref_t<T>>::value;

template <class T>
struct is_out_marker : std::false_type {};

template <class OwnershipReturn>
struct is_out_marker<out<OwnershipReturn>> : std::true_type {};

template <class T>
inline constexpr bool is_out_marker_v = is_out_marker<remove_cvref_t<T>>::value;

template <class T>
inline constexpr bool is_objc_cpp_object_return_v = has_objc_arg_v<T>;

template <class T>
inline constexpr bool is_invalid_boundary_parameter_v =
    is_object_wrapper_v<T> || is_objc_ownership_return_v<T>;

template <class T>
decltype(auto) unwrap_arg(T &&x) noexcept {
  if constexpr (has_objc_arg_v<T>) {
    return x.objc_arg();
  } else {
    return std::forward<T>(x);
  }
}

// The raw, ABI-level type a C++ argument lowers to for objc_msgSend and block
// dispatch: facades and owning wrappers collapse to their void* objc_arg(),
// scalars pass through unchanged. unwrap_arg() yields a value of this type.
template <class T>
using raw_arg_t = std::decay_t<decltype(unwrap_arg(std::declval<T>()))>;

template <class T>
inline constexpr bool is_objc_object_pointer_v =
    std::is_pointer_v<std::remove_reference_t<T>> &&
    std::is_same_v<std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<T>>>,
                   std::remove_pointer_t<::id>>;

template <class T>
inline constexpr bool is_void_pointer_v =
    std::is_pointer_v<std::remove_reference_t<T>> &&
    std::is_void_v<std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<T>>>>;

template <class Raw, class Value>
Raw coerce_declared_objc_raw(Value &&value) noexcept {
  if constexpr (std::is_same_v<Raw, void *> &&
                (is_objc_object_pointer_v<Value> || is_void_pointer_v<Value>)) {
    return const_cast<void *>(static_cast<const void *>(value));
  } else {
    return static_cast<Raw>(std::forward<Value>(value));
  }
}

template <class Spec, class CallArg>
struct block_arg_signature_matches : std::false_type {};

template <class SpecSignature, class CallSignature>
struct block_arg_signature_matches<block<SpecSignature>, block<CallSignature>>
    : std::is_same<SpecSignature, CallSignature> {};

template <class Spec, class CallArg>
struct is_method_arg_compatible : std::true_type {};

template <class Signature, class CallArg>
struct is_method_arg_compatible<block<Signature>, CallArg>
    : std::bool_constant<
          std::is_null_pointer_v<remove_cvref_t<CallArg>> ||
          block_arg_signature_matches<block<Signature>, remove_cvref_t<CallArg>>::value> {};

template <class Spec, class CallArg>
inline constexpr bool is_method_arg_compatible_v =
    is_method_arg_compatible<remove_cvref_t<Spec>, remove_cvref_t<CallArg>>::value;

template <class Spec, class CallArg>
raw_arg_t<Spec> coerce_arg_to_declared_raw(CallArg &&arg) noexcept {
  static_assert(is_method_arg_compatible_v<Spec, CallArg>,
                "objc::method block parameters require objc::block<Sig> with the same "
                "signature, or nullptr");

  using raw_type = raw_arg_t<Spec>;
  if constexpr (!is_method_arg_compatible_v<Spec, CallArg>) {
    return raw_type{};
  } else if constexpr (has_objc_arg_v<Spec>) {
    return coerce_declared_objc_raw<raw_type>(unwrap_arg(std::forward<CallArg>(arg)));
  } else {
    return static_cast<raw_type>(unwrap_arg(std::forward<CallArg>(arg)));
  }
}

template <typename Ret, typename... Args>
inline Ret send_dispatch(const void *self, SEL op, Args... args) {
  if constexpr (uses_fp2ret_v<Ret>) {
    return call_fp2ret<Ret>(self, op, args...);
  } else if constexpr (uses_fpret_v<Ret>) {
    return call_fpret<Ret>(self, op, args...);
  } else if constexpr (indirect_return_v<Ret>) {
    if (self == nullptr) return Ret{};
    return call_indirect<Ret>(self, op, args...);
  } else {
    return call_msgSend<Ret>(self, op, args...);
  }
}

}  // namespace detail

template <typename Ret, typename... Args>
inline Ret send(const void *self, SEL op, Args &&...args) {
  static_assert(!detail::is_objc_cpp_object_return_v<Ret>,
                "objc::send<Ret>: Ret must be a raw ABI return type. Use ::id for raw "
                "object returns, or objc::method<retained<T> / not_retained<T> / "
                "autoreleased<T>> for ownership-aware object returns.");
  static_assert(!detail::is_objc_ownership_return_v<Ret>,
                "objc::send<retained<T> / not_retained<T> / autoreleased<T>> is not "
                "valid. Ownership markers are only interpreted by objc::method.");
  static_assert(!detail::is_out_marker_v<Ret>,
                "objc::send<objc::out<...>> is not valid. objc::out<...> is only "
                "interpreted by objc::method parameters.");
  return detail::send_dispatch<Ret>(self, op, detail::unwrap_arg(std::forward<Args>(args))...);
}

// ===========================================================================
// handle and borrowed typed facades
// ===========================================================================

class OBJC_GSL_POINTER handle {
 public:
  constexpr handle() noexcept = default;
  constexpr handle(std::nullptr_t) noexcept {}
  explicit constexpr handle(const_id p OBJC_LIFETIMEBOUND) noexcept : ptr_(p) {}

  const_id ptr() const noexcept OBJC_LIFETIMEBOUND { return ptr_; }
  id mut_ptr() const noexcept OBJC_LIFETIMEBOUND { return const_cast<::id>(ptr_); }

  void *objc_arg() const noexcept OBJC_LIFETIMEBOUND { return mut_ptr(); }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  Class runtime_class() const { return object_getClass(reinterpret_cast<::id>(mut_ptr())); }

  bool is_kind_of(Class cls) const {
    return send<BOOL>(ptr(), OBJC_SEL("isKindOfClass:"), cls) != NO;
  }

  bool responds_to(SEL sel) const {
    return send<BOOL>(ptr(), OBJC_SEL("respondsToSelector:"), sel) != NO;
  }

  friend bool operator==(handle a, handle b) noexcept { return a.ptr_ == b.ptr_; }

  friend bool operator!=(handle a, handle b) noexcept { return !(a == b); }

 protected:
  const_id ptr_ = nil;
};

// Non-owning statically typed Objective-C values are represented directly by
// facade types deriving from handle, such as NSString or protocol<P>.
//
//   objc::NSString s = owner;  // lvalue owner -> borrowed facade
//   s.utf8();
//
// There is intentionally no separate borrowed-wrapper template in this variation.
namespace detail {

inline const void *recv_ptr(const void *p) noexcept { return p; }
inline const void *recv_ptr(void *p) noexcept { return p; }
inline const void *recv_ptr(std::nullptr_t) noexcept { return nullptr; }

template <class T>
inline const void *recv_ptr(T &&x) noexcept {
  if constexpr (has_objc_arg_v<T>) {
    return x.objc_arg();
  } else if constexpr (std::is_pointer_v<std::remove_reference_t<T>>) {
    return x;
  } else {
    static_assert(always_false_v<std::remove_cv_t<std::remove_reference_t<T>>>,
                  "receiver must be a raw pointer or expose objc_arg()");
  }
}

}  // namespace detail

// ===========================================================================
// Owning object<T>
// ===========================================================================

struct retain_t {
  explicit retain_t() = default;
};

struct adopt_t {
  explicit adopt_t() = default;
};

inline constexpr retain_t retain_ref{};
inline constexpr adopt_t adopt_ref{};

enum class return_value_policy { retained, not_retained, autoreleased };

template <class T = handle>
class OBJC_GSL_OWNER object {
 public:
  static_assert(std::is_base_of_v<handle, T>,
                "objc::object<T>: T must be an Objective-C facade deriving from objc::handle");

  using element_type = T;

  object() noexcept = default;
  object(std::nullptr_t) noexcept : value_(nullptr) {}

  object(retain_t, const_id p) : value_(p) { retain_if_nonnull(); }

  object(adopt_t, const_id p) noexcept : value_(p) {}

  object(retain_t, T v) : object(::objc::retain_ref, v.ptr()) {}

  object(const object &other) : value_(other.value_) { retain_if_nonnull(); }

  object(object &&other) noexcept : value_(other.value_) { other.value_ = T(nullptr); }

  template <class U, std::enable_if_t<std::is_base_of_v<T, U> && !std::is_same_v<T, U>, int> = 0>
  object(const object<U> &other) : value_(T(other.ptr())) {
    retain_if_nonnull();
  }

  template <class U, std::enable_if_t<std::is_base_of_v<T, U> && !std::is_same_v<T, U>, int> = 0>
  object(object<U> &&other) noexcept : value_(T(other.release())) {}

  ~object() { release_if_nonnull(); }

  object &operator=(const object &other) {
    if (this == &other) return *this;

    const_id new_ptr = other.ptr();
    if (new_ptr) send<void>(new_ptr, OBJC_SEL("retain"));

    const_id old_ptr = ptr();
    value_ = T(new_ptr);

    if (old_ptr) send<void>(old_ptr, OBJC_SEL("release"));
    return *this;
  }

  object &operator=(object &&other) noexcept {
    if (this == &other) return *this;

    const_id old_ptr = ptr();
    value_ = other.value_;
    other.value_ = T(nullptr);

    if (old_ptr) send<void>(old_ptr, OBJC_SEL("release"));
    return *this;
  }

  object &operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  const_id ptr() const noexcept OBJC_LIFETIMEBOUND { return value_.ptr(); }
  id mut_ptr() const noexcept OBJC_LIFETIMEBOUND { return value_.mut_ptr(); }
  void *objc_arg() const noexcept OBJC_LIFETIMEBOUND { return value_.objc_arg(); }

  explicit operator bool() const noexcept { return static_cast<bool>(value_); }

  const T *operator->() const & noexcept OBJC_LIFETIMEBOUND { return &value_; }
  T *operator->() & noexcept OBJC_LIFETIMEBOUND { return &value_; }
  const T *operator->() const && = delete;
  T *operator->() && = delete;

  T ref() const & noexcept OBJC_LIFETIMEBOUND { return value_; }
  T ref() && = delete;

  operator T() const & noexcept OBJC_LIFETIMEBOUND { return ref(); }
  operator T() && = delete;

  template <class U, std::enable_if_t<std::is_base_of_v<U, T> && !std::is_same_v<U, T>, int> = 0>
  U ref() const & noexcept OBJC_LIFETIMEBOUND {
    return U(ptr());
  }

  template <class U, std::enable_if_t<std::is_base_of_v<U, T> && !std::is_same_v<U, T>, int> = 0>
  U ref() && = delete;

  template <class U, std::enable_if_t<std::is_base_of_v<U, T> && !std::is_same_v<U, T>, int> = 0>
  operator U() const & noexcept OBJC_LIFETIMEBOUND {
    return ref<U>();
  }

  template <class U, std::enable_if_t<std::is_base_of_v<U, T> && !std::is_same_v<U, T>, int> = 0>
  operator U() && = delete;

  void reset() noexcept {
    const_id old_ptr = ptr();
    value_ = T(nullptr);

    if (old_ptr) send<void>(old_ptr, OBJC_SEL("release"));
  }

  void reset(retain_t, const_id p) {
    if (p) send<void>(p, OBJC_SEL("retain"));

    const_id old_ptr = ptr();
    value_ = T(p);

    if (old_ptr) send<void>(old_ptr, OBJC_SEL("release"));
  }

  void reset(adopt_t, const_id p) noexcept {
    const_id old_ptr = ptr();
    value_ = T(p);

    if (old_ptr) send<void>(old_ptr, OBJC_SEL("release"));
  }

  void reset(retain_t, T v) { reset(::objc::retain_ref, v.ptr()); }

  [[nodiscard]] ::id release() noexcept {
    ::id p = mut_ptr();
    value_ = T(nullptr);
    return p;
  }

 private:
  void retain_if_nonnull() {
    if (ptr()) send<void>(ptr(), OBJC_SEL("retain"));
  }

  void release_if_nonnull() noexcept {
    if (ptr()) send<void>(ptr(), OBJC_SEL("release"));
  }

  T value_{nullptr};
};

template <class T>
[[nodiscard]] object<T> retain(const_id p) {
  return object<T>(retain_ref, p);
}

template <class T>
[[nodiscard]] object<T> retain(T v) {
  static_assert(std::is_base_of_v<handle, T>,
                "objc::retain<T>(T): T must derive from objc::handle");
  return object<T>(retain_ref, v);
}

template <class T>
[[nodiscard]] object<T> adopt(const_id p) noexcept {
  return object<T>(adopt_ref, p);
}

template <class T>
[[nodiscard]] T ref(const_id p OBJC_LIFETIMEBOUND) noexcept {
  static_assert(std::is_base_of_v<handle, T>, "objc::ref<T>: T must derive from objc::handle");
  return T(p);
}

template <class T>
[[nodiscard]] T cast(const handle &h OBJC_LIFETIMEBOUND) noexcept {
  static_assert(std::is_base_of_v<handle, T>, "objc::cast<T>: T must derive from objc::handle");
  return T(h.ptr());
}

template <class T>
[[nodiscard]] object<T> retain_cast(const handle &h) {
  return retain<T>(h.ptr());
}

template <class T>
[[nodiscard]] object<T> adopt_cast(const handle &h OBJC_LIFETIMEBOUND) noexcept {
  return adopt<T>(h.ptr());
}

template <class T>
[[nodiscard]] T dyn_cast(const handle &h OBJC_LIFETIMEBOUND) noexcept {
  static_assert(std::is_base_of_v<handle, T>, "objc::dyn_cast<T>: T must derive from objc::handle");
  if (!h) return {};

  T v(h.ptr());

  if (send<BOOL>(v.ptr(), OBJC_SEL("isKindOfClass:"), T::cls()) == NO) {
    return {};
  }

  return v;
}

template <class T>
[[nodiscard]] object<T> dyn_retain(const handle &h) {
  auto v = dyn_cast<T>(h);
  return v ? retain<T>(v) : object<T>{};
}

// ===========================================================================
// Protocol wrapper — model id<P>
// ===========================================================================

template <class P>
class protocol : public handle {
 public:
  using handle::handle;

  static Protocol *proto() { return P::proto(); }
};

template <class P>
[[nodiscard]] protocol<P> dyn_protocol_cast(const handle &h OBJC_LIFETIMEBOUND) noexcept {
  if (!h) return {};

  protocol<P> v(h.ptr());

  BOOL ok = send<BOOL>(v.ptr(), OBJC_SEL("conformsToProtocol:"), P::proto());

  if (ok == NO) return {};
  return v;
}

template <class P>
[[nodiscard]] object<protocol<P>> dyn_protocol_retain(const handle &h) {
  auto v = dyn_protocol_cast<P>(h);
  return v ? retain<protocol<P>>(v) : object<protocol<P>>{};
}

// ===========================================================================
// Autorelease
// ===========================================================================

template <class T>
[[nodiscard]] T autorelease(object<T> &&obj) {
  ::id p = obj.release();

  if (p) {
    send<void>(p, OBJC_SEL("autorelease"));
  }

  return T(p);
}

// ===========================================================================
// alloc/init helpers
// ===========================================================================
//
// alloc returns +1, but it is usually an intermediate allocation state, not a
// finished object result. Do not wrap alloc as object<T> and then call init
// as another retained<T> method; that creates two RAII owners for the same +1
// claim when init returns self.
//
// Instead, keep the allocated pointer raw until init returns the final
// object, then adopt that returned pointer exactly once.
//
// If an initializer returns nil, this assumes the normal Objective-C
// initializer convention: the initializer is responsible for releasing the
// allocated receiver before returning nil.

// Optional explicit allocation state, useful if you want to split alloc/init
// without accidentally creating two object<T> owners.
template <class T>
class allocated {
 public:
  allocated() noexcept = default;
  explicit allocated(::id p) noexcept : ptr_(p) {}

  allocated(const allocated &) = delete;
  allocated &operator=(const allocated &) = delete;

  allocated(allocated &&other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}

  allocated &operator=(allocated &&other) noexcept {
    if (this == &other) return *this;

    reset();
    ptr_ = std::exchange(other.ptr_, nullptr);
    return *this;
  }

  ~allocated() { reset(); }

  ::id get() const noexcept { return ptr_; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  void reset() noexcept {
    if (ptr_) {
      send<void>(ptr_, OBJC_SEL("release"));
      ptr_ = nullptr;
    }
  }

  ::id release() noexcept { return std::exchange(ptr_, nullptr); }

 private:
  ::id ptr_ = nullptr;
};

template <class T>
[[nodiscard]] allocated<T> alloc() {
  return allocated<T>(send<::id>(T::cls(), OBJC_SEL("alloc")));
}

template <class T, class... Args>
[[nodiscard]] object<T> init(allocated<T> &&a, SEL init_sel, Args... args) {
  ::id raw = a.release();
  ::id initialized = send<::id>(raw, init_sel, args...);
  return object<T>(adopt_ref, initialized);
}

template <class T, class... Args>
[[nodiscard]] object<T> alloc_init(SEL init_sel, Args... args) {
  return init<T>(alloc<T>(), init_sel, args...);
}

// ===========================================================================
// Ownership-qualified method return markers
// ===========================================================================

namespace detail {

template <class T, class = void>
struct is_complete : std::false_type {};

template <class T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <class T>
inline constexpr bool is_complete_objc_facade_v = [] {
  static_assert(is_complete<T>::value, "T must be complete");

  return std::is_base_of_v<handle, T>;
}();

}  // namespace detail

template <class T>
struct retained {
  static_assert(detail::is_complete_objc_facade_v<T>,
                "objc::retained<T>: T must be a complete Objective-C facade deriving "
                "from objc::handle");

  using element_type = T;
};

template <class T>
struct not_retained {
  static_assert(detail::is_complete_objc_facade_v<T>,
                "objc::not_retained<T>: T must be a complete Objective-C facade deriving "
                "from objc::handle");

  using element_type = T;
};

template <class T>
struct autoreleased {
  static_assert(detail::is_complete_objc_facade_v<T>,
                "objc::autoreleased<T>: T must be a complete Objective-C facade deriving "
                "from objc::handle");

  using element_type = T;
};

// Marker for Objective-C object out-parameters such as NSError **.
//
// The ownership policy is part of the method signature, while the call site
// passes an object<T>& destination:
//
//   object<NSError> error;
//   method<retained<Foo>(Bar, out<autoreleased<NSError>>)> m(...);
//   object<Foo> foo = m(receiver, bar, error);
//
// The method creates the raw T ** slot internally and assigns the destination
// using the policy encoded by out<...>.
template <class OwnershipReturn>
struct out {
  using ownership_type = OwnershipReturn;
};

namespace detail {

template <class T>
struct is_valid_out_ownership_return : std::false_type {};

template <class T>
struct is_valid_out_ownership_return<retained<T>> : is_objc_ownership_return<retained<T>> {};

template <class T>
struct is_valid_out_ownership_return<not_retained<T>> : is_objc_ownership_return<not_retained<T>> {
};

template <class T>
struct is_valid_out_ownership_return<autoreleased<T>> : is_objc_ownership_return<autoreleased<T>> {
};

template <class T>
inline constexpr bool is_valid_out_ownership_return_v =
    is_valid_out_ownership_return<remove_cvref_t<T>>::value;

template <class T>
struct is_valid_out_parameter : std::false_type {};

template <class OwnershipReturn>
struct is_valid_out_parameter<out<OwnershipReturn>>
    : std::bool_constant<is_valid_out_ownership_return_v<OwnershipReturn>> {};

template <class T>
inline constexpr bool is_valid_out_parameter_v = is_valid_out_parameter<remove_cvref_t<T>>::value;

template <class T>
inline constexpr bool is_invalid_method_parameter_v =
    is_invalid_boundary_parameter_v<T> || (is_out_marker_v<T> && !is_valid_out_parameter_v<T>);

template <class T>
inline constexpr bool is_invalid_block_parameter_v =
    is_invalid_boundary_parameter_v<T> || is_out_marker_v<T>;

template <class Ret>
struct objc_return_traits;

// +1 result. Adopt; do not retain.
template <class T>
struct objc_return_traits<retained<T>> {
  using element_type = T;
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  static cpp_return_type convert(raw_return_type p) noexcept { return adopt<T>(p); }

  static constexpr bool may_return_ref = false;
  static constexpr bool may_return_unsafe_ref = false;
};

// +0 result. Retain into C++ ownership by default.
template <class T>
struct objc_return_traits<not_retained<T>> {
  using element_type = T;
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  static cpp_return_type convert(raw_return_type p) { return retain<T>(p); }

  static T make_ref(raw_return_type p) noexcept { return T(p); }

  static constexpr bool may_return_ref = true;
  static constexpr bool may_return_unsafe_ref = true;
};

// +0 autoreleased result. Retain into C++ ownership by default.
//
// This intentionally uses plain retain(), not an optimized
// objc_retainAutoreleasedReturnValue path.
template <class T>
struct objc_return_traits<autoreleased<T>> {
  using element_type = T;
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  static cpp_return_type convert(raw_return_type p) { return retain<T>(p); }

  static T make_ref(raw_return_type p) noexcept { return T(p); }

  static constexpr bool may_return_ref = false;
  static constexpr bool may_return_unsafe_ref = true;
};

template <class T>
struct is_out_arg : std::false_type {};

template <class OwnershipReturn>
struct is_out_arg<out<OwnershipReturn>> : std::true_type {};

template <class T>
inline constexpr bool is_out_arg_v =
    is_out_arg<std::remove_cv_t<std::remove_reference_t<T>>>::value;

template <class Spec, class CallArg, bool IsOut = is_out_arg_v<Spec>>
class method_arg_slot;

// Normal Objective-C message argument. Stores the unwrapped raw argument value
// used for the objc_msgSend call.
template <class Spec, class CallArg>
class method_arg_slot<Spec, CallArg, false> {
 public:
  using raw_type = raw_arg_t<Spec>;

  explicit method_arg_slot(CallArg arg) noexcept
      : raw_(coerce_arg_to_declared_raw<Spec>(std::forward<CallArg>(arg))) {}

  raw_type raw_arg() noexcept { return raw_; }

  void commit() noexcept {}

 private:
  raw_type raw_;
};

// Objective-C object out-parameter. The call site passes object<T>&, while the
// method signature says out<retained<T>>, out<not_retained<T>>, or
// out<autoreleased<T>>.
template <class OwnershipReturn, class CallArg>
class method_arg_slot<out<OwnershipReturn>, CallArg, true> {
 public:
  using traits = objc_return_traits<OwnershipReturn>;
  using element_type = typename traits::element_type;
  using raw_type = typename traits::raw_return_type;
  using destination_type = object<element_type>;

  explicit method_arg_slot(CallArg arg) noexcept : dest_(&arg), raw_(nil) {
    static_assert(std::is_lvalue_reference_v<CallArg>,
                  "objc::out<...> method arguments require an lvalue destination");
    static_assert(
        std::is_same_v<std::remove_cv_t<std::remove_reference_t<CallArg>>, destination_type>,
        "objc::out<ownership<T>> method arguments require object<T>&");
  }

  raw_type *raw_arg() noexcept { return &raw_; }

  void commit() { *dest_ = traits::convert(raw_); }

 private:
  destination_type *dest_;
  raw_type raw_;
};

template <class SlotTuple, std::size_t... I>
void commit_out_slots(SlotTuple &slots, std::index_sequence<I...>) {
  (std::get<I>(slots).commit(), ...);
}

}  // namespace detail

// ===========================================================================
// selector<Signature> — raw typed selector
// ===========================================================================

template <class Signature>
class selector;

template <class Ret, class... Args>
class selector<Ret(Args...)> {
 public:
  explicit selector(const char *name) noexcept : sel_(sel_registerName(name)) {}
  explicit selector(SEL s) noexcept : sel_(s) {}

  SEL get() const noexcept { return sel_; }

  template <class Self>
  Ret operator()(Self &&self, Args... args) const {
    return send<Ret>(detail::recv_ptr(std::forward<Self>(self)), sel_, args...);
  }

 private:
  SEL sel_;
};

// ===========================================================================
// method<Signature> — ownership-aware typed selector
// ===========================================================================

template <class Signature>
class method;

template <class Ret, class... Args>
class method<Ret(Args...)> {
  static_assert(!detail::is_objc_cpp_object_return_v<Ret>,
                "objc::method return type must not be an Objective-C facade, "
                "objc::object<T>, or objc::block<Sig>. Use objc::retained<T>, "
                "objc::not_retained<T>, or objc::autoreleased<T> for Objective-C object "
                "returns.");
  static_assert(!detail::is_out_marker_v<Ret>,
                "objc::method return type must not be objc::out<...>. Use "
                "objc::out<retained<T> / not_retained<T> / autoreleased<T>> only as a "
                "method parameter.");
  static_assert((!detail::is_invalid_method_parameter_v<Args> && ...),
                "objc::method parameters must not be objc::object<T>, Objective-C "
                "ownership return markers, or invalid objc::out<...>. Use borrowed "
                "facades such as objc::NSString, objc::handle, objc::block<Sig>, or "
                "objc::out<retained<T> / not_retained<T> / autoreleased<T>>.");

 public:
  explicit method(const char *name) noexcept : sel_(sel_registerName(name)) {}
  explicit method(SEL sel) noexcept : sel_(sel) {}

  SEL get() const noexcept { return sel_; }

  template <class Self, class... CallArgs>
  decltype(auto) operator()(Self &&self, CallArgs &&...call_args) const {
    static_assert(sizeof...(CallArgs) == sizeof...(Args),
                  "objc::method call has the wrong number of explicit arguments");
    return call_impl(detail::recv_ptr(std::forward<Self>(self)),
                     std::forward<CallArgs>(call_args)...);
  }

  template <class Self, class... CallArgs>
  auto ref(Self &&self, CallArgs &&...call_args) const {
    static_assert(sizeof...(CallArgs) == sizeof...(Args),
                  "objc::method::ref call has the wrong number of explicit arguments");
    static_assert(detail::is_objc_ownership_return_v<Ret>,
                  "method::ref() is only meaningful for Objective-C object "
                  "return markers such as objc::not_retained<T>");

    using traits = detail::objc_return_traits<Ret>;

    static_assert(traits::may_return_ref,
                  "method::ref() is only allowed for objc::not_retained<T>. "
                  "For retained<T>, use operator() to adopt the +1 result. "
                  "For autoreleased<T>, use operator() to retain the result, "
                  "or unsafe_ref() if you really want a pool-bound ref.");

    return ref_impl<traits>(detail::recv_ptr(std::forward<Self>(self)),
                            std::forward<CallArgs>(call_args)...);
  }

  template <class Self, class... CallArgs>
  auto unsafe_ref(Self &&self, CallArgs &&...call_args) const {
    static_assert(sizeof...(CallArgs) == sizeof...(Args),
                  "objc::method::unsafe_ref call has the wrong number of explicit arguments");
    static_assert(detail::is_objc_ownership_return_v<Ret>,
                  "method::unsafe_ref() is only meaningful for Objective-C "
                  "object return markers");

    using traits = detail::objc_return_traits<Ret>;

    static_assert(traits::may_return_unsafe_ref,
                  "method::unsafe_ref() is not allowed for objc::retained<T>, "
                  "because it would drop a +1 result without releasing it.");

    return ref_impl<traits>(detail::recv_ptr(std::forward<Self>(self)),
                            std::forward<CallArgs>(call_args)...);
  }

 private:
  template <class SelfPtr, class... CallArgs>
  decltype(auto) call_impl(SelfPtr self, CallArgs &&...call_args) const {
    auto slots = std::tuple<detail::method_arg_slot<Args, CallArgs &&>...>(
        detail::method_arg_slot<Args, CallArgs &&>(std::forward<CallArgs>(call_args))...);

    return call_with_slots(self, slots, std::index_sequence_for<Args...>{});
  }

  template <class SelfPtr, class SlotTuple, std::size_t... I>
  decltype(auto) call_with_slots(SelfPtr self, SlotTuple &slots, std::index_sequence<I...>) const {
    if constexpr (detail::is_objc_ownership_return_v<Ret>) {
      using traits = detail::objc_return_traits<Ret>;
      using RawRet = typename traits::raw_return_type;

      RawRet raw = send<RawRet>(self, sel_, std::get<I>(slots).raw_arg()...);
      auto result = traits::convert(raw);
      detail::commit_out_slots(slots, std::index_sequence<I...>{});
      return result;
    } else if constexpr (std::is_void_v<Ret>) {
      send<void>(self, sel_, std::get<I>(slots).raw_arg()...);
      detail::commit_out_slots(slots, std::index_sequence<I...>{});
    } else {
      Ret r = send<Ret>(self, sel_, std::get<I>(slots).raw_arg()...);
      detail::commit_out_slots(slots, std::index_sequence<I...>{});
      return r;
    }
  }

  template <class Traits, class SelfPtr, class... CallArgs>
  auto ref_impl(SelfPtr self, CallArgs &&...call_args) const {
    auto slots = std::tuple<detail::method_arg_slot<Args, CallArgs &&>...>(
        detail::method_arg_slot<Args, CallArgs &&>(std::forward<CallArgs>(call_args))...);

    return ref_with_slots<Traits>(self, slots, std::index_sequence_for<Args...>{});
  }

  template <class Traits, class SelfPtr, class SlotTuple, std::size_t... I>
  auto ref_with_slots(SelfPtr self, SlotTuple &slots, std::index_sequence<I...>) const {
    using RawRet = typename Traits::raw_return_type;
    RawRet raw = send<RawRet>(self, sel_, std::get<I>(slots).raw_arg()...);
    detail::commit_out_slots(slots, std::index_sequence<I...>{});
    return Traits::make_ref(raw);
  }

  SEL sel_;
};

// ===========================================================================
// Blocks — C++ callables as Objective-C blocks
// ===========================================================================

namespace detail {

// Compiler-set bits of the block_layout flags word.
//
// BLOCK_HAS_COPY_DISPOSE is the load-bearing one: it tells the runtime the
// descriptor carries copy/dispose helpers, so _Block_copy / _Block_release
// invoke them — this is how the captured C++ closure's copy ctor / dtor run.
//
// BLOCK_HAS_CTOR ("helpers have C++ code") is set by Clang when a block captures
// C++ objects with non-trivial copy/destroy. The modern runtime ignores it:
// copy/dispose dispatch is gated solely on BLOCK_HAS_COPY_DISPOSE. It only ever
// mattered to the long-removed Objective-C garbage collector (which could skip a
// copy helper for plain object captures, but not when C++ ctors/dtors had to
// run). We set it for fidelity with the compiler, not because anything reads it.
enum block_flags : int {
  BLOCK_HAS_COPY_DISPOSE = 1 << 25,
  BLOCK_HAS_CTOR = 1 << 26,
};

struct block_descriptor {
  unsigned long reserved;
  unsigned long size;
  void (*copy)(void *dst, const void *src);
  void (*dispose)(const void *src);
};

struct block_header {
  void *isa;
  int flags;
  int reserved;
  void *invoke;
  block_descriptor *descriptor;
};

// A block literal is a block_header immediately followed by the captured C++
// callable. _Block_copy() copies descriptor->size (== sizeof(block_literal))
// bytes, and the runtime reads the header at offset 0, so `header` must stay the
// first member. Embedding block_header — rather than re-listing its fields —
// keeps the two layouts from drifting apart.
template <class F, class Ret, class... Args>
struct block_literal {
  block_header header;
  F fn;
};

// Inverse of unwrap_arg, for the block trampoline. The Objective-C runtime hands
// back each id-typed parameter as a raw pointer; we rewrap it into the facade the
// C++ callable declared (scalars arrive unchanged). This is the same facade<->id
// boundary as a message send, run in the opposite direction.
template <class Param>
Param wrap_block_arg(raw_arg_t<Param> raw) noexcept {
  if constexpr (std::is_base_of_v<handle, Param>) {
    return Param(static_cast<const_id>(raw));
  } else {
    static_assert(!has_objc_arg_v<Param>,
                  "objc::block parameters must be borrowed facades (deriving "
                  "from objc::handle) or scalars, not owning wrappers");
    return raw;
  }
}

template <class T, class Result>
object<T> block_result_to_owner(Result &&result) {
  using R = remove_cvref_t<Result>;
  static_assert(std::is_base_of_v<handle, T>,
                "objc::block object return markers require facade types");

  if constexpr (std::is_null_pointer_v<R>) {
    return {};
  } else if constexpr (is_object_wrapper_v<R>) {
    using U = typename is_object_wrapper<R>::element_type;
    static_assert(std::is_base_of_v<T, U>, "objc::block object return owner must be object<T> or "
                                           "object<U> where U derives from T");
    return object<T>(std::forward<Result>(result));
  } else if constexpr (std::is_base_of_v<handle, R>) {
    static_assert(std::is_base_of_v<T, R>,
                  "objc::block object return facade must be T or derive from T");
    return retain<T>(T(result.ptr()));
  } else {
    static_assert(always_false_v<R>, "objc::block object return markers require object<T>, a "
                                     "borrowed facade, or nullptr");
  }
}

template <class T, class Result>
::id block_borrowed_result_to_raw(Result &&result) noexcept {
  using R = remove_cvref_t<Result>;
  static_assert(std::is_base_of_v<handle, T>,
                "objc::block object return markers require facade types");

  if constexpr (std::is_null_pointer_v<R>) {
    return nil;
  } else if constexpr (is_object_wrapper_v<R>) {
    static_assert(always_false_v<R>, "objc::block<not_retained<T>()> callbacks must return a "
                                     "borrowed facade, not object<T>");
  } else if constexpr (std::is_base_of_v<handle, R>) {
    static_assert(std::is_base_of_v<T, R>,
                  "objc::block object return facade must be T or derive from T");
    return result.mut_ptr();
  } else {
    static_assert(always_false_v<R>, "objc::block<not_retained<T>()> callbacks must return a "
                                     "borrowed facade or nullptr");
  }
}

template <class Ret>
struct block_return_traits {
  static_assert(!is_objc_cpp_object_return_v<Ret>,
                "objc::block return type must not be an Objective-C facade, "
                "objc::object<T>, or objc::block<Sig>. Use objc::retained<T>, "
                "objc::not_retained<T>, or objc::autoreleased<T> for Objective-C object "
                "returns, or ::id for raw object returns.");
  static_assert(!is_out_marker_v<Ret>,
                "objc::block return type must not be objc::out<...>. objc::out<...> is "
                "only valid as an objc::method parameter.");

  using raw_return_type = Ret;
  using cpp_return_type = Ret;

  template <class F, class... CallArgs>
  static raw_return_type call(F &fn, CallArgs &&...call_args) {
    return fn(std::forward<CallArgs>(call_args)...);
  }

  static cpp_return_type from_raw(raw_return_type raw) { return raw; }
};

template <>
struct block_return_traits<void> {
  using raw_return_type = void;
  using cpp_return_type = void;

  template <class F, class... CallArgs>
  static void call(F &fn, CallArgs &&...call_args) {
    fn(std::forward<CallArgs>(call_args)...);
  }
};

template <class T>
struct block_return_traits<retained<T>> {
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  template <class F, class... CallArgs>
  static raw_return_type call(F &fn, CallArgs &&...call_args) {
    auto result = fn(std::forward<CallArgs>(call_args)...);
    return block_result_to_owner<T>(std::move(result)).release();
  }

  static cpp_return_type from_raw(raw_return_type raw) noexcept { return adopt<T>(raw); }
};

template <class T>
struct block_return_traits<not_retained<T>> {
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  template <class F, class... CallArgs>
  static raw_return_type call(F &fn, CallArgs &&...call_args) noexcept {
    auto result = fn(std::forward<CallArgs>(call_args)...);
    return block_borrowed_result_to_raw<T>(result);
  }

  static cpp_return_type from_raw(raw_return_type raw) { return retain<T>(raw); }
};

template <class T>
struct block_return_traits<autoreleased<T>> {
  using raw_return_type = ::id;
  using cpp_return_type = object<T>;

  template <class F, class... CallArgs>
  static raw_return_type call(F &fn, CallArgs &&...call_args) {
    auto result = fn(std::forward<CallArgs>(call_args)...);
    object<T> owner = block_result_to_owner<T>(std::move(result));
    ::id raw = owner.release();
    if (raw) send<void>(raw, OBJC_SEL("autorelease"));
    return raw;
  }

  static cpp_return_type from_raw(raw_return_type raw) { return retain<T>(raw); }
};

// The invoke trampoline runs at the raw ABI: it declares exactly the
// pointer/scalar parameters the runtime passes (raw_arg_t<Args>...), then rewraps
// each into the facade the callable expects. Nothing here depends on a facade's
// calling convention matching id's.
template <class Lit, class Ret, class... Args>
typename block_return_traits<Ret>::raw_return_type block_invoke(Lit *self,
                                                                raw_arg_t<Args>... raw) noexcept {
  using traits = block_return_traits<Ret>;
  using RawRet = typename traits::raw_return_type;

  if constexpr (std::is_void_v<RawRet>) {
    traits::call(self->fn, wrap_block_arg<Args>(raw)...);
  } else {
    return traits::call(self->fn, wrap_block_arg<Args>(raw)...);
  }
}

// Copy/dispose run the captured C++ closure's own copy constructor and
// destructor. A Clang-emitted Objective-C block instead stores raw captured
// id / block / __block slots and calls _Block_object_assign /
// _Block_object_dispose so the runtime retains/releases each; we don't, because
// our only capture is the opaque closure `fn`. Any Objective-C object it holds
// is owned through a C++ wrapper (objc::object<T>, objc::block) whose
// copy/destructor does the retain/release, so _Block_object_assign/_dispose are
// neither needed nor correct here — `fn` is a C++ value, not an id slot.
template <class Lit, class F>
void block_copy_helper(void *dst, const void *src) noexcept {
  auto *d = static_cast<Lit *>(dst);
  auto *s = static_cast<const Lit *>(src);
  ::new (static_cast<void *>(&d->fn)) F(s->fn);
}

template <class Lit, class F>
void block_dispose_helper(const void *src) {
  static_cast<const Lit *>(src)->fn.~F();
}

template <class Lit, class F>
block_descriptor *block_desc() {
  static block_descriptor d = {
      .reserved = 0,
      .size = sizeof(Lit),
      .copy = &block_copy_helper<Lit, F>,
      .dispose = &block_dispose_helper<Lit, F>,
  };

  return &d;
}

template <class F>
constexpr int block_flags_for() noexcept {
  // BLOCK_HAS_COPY_DISPOSE makes the runtime call our copy/dispose helpers;
  // BLOCK_HAS_CTOR only mirrors Clang (see the enum) and is inert today.
  int f = BLOCK_HAS_COPY_DISPOSE;

  if constexpr (!std::is_trivially_copy_constructible_v<F> ||
                !std::is_trivially_destructible_v<F>) {
    f |= BLOCK_HAS_CTOR;
  }

  return f;
}

}  // namespace detail

template <class Signature>
class block;

template <class Ret, class... Args>
class block<Ret(Args...)> {
  static_assert(!detail::is_objc_cpp_object_return_v<Ret>,
                "objc::block return type must not be an Objective-C facade, "
                "objc::object<T>, or objc::block<Sig>. Use objc::retained<T>, "
                "objc::not_retained<T>, or objc::autoreleased<T> for Objective-C object "
                "returns, or ::id for raw object returns.");
  static_assert(!detail::is_out_marker_v<Ret>,
                "objc::block return type must not be objc::out<...>. objc::out<...> is "
                "only valid as an objc::method parameter.");
  static_assert((!detail::is_invalid_block_parameter_v<Args> && ...),
                "objc::block parameters must not be objc::object<T>, objc::out<...>, or "
                "Objective-C ownership return markers. Use borrowed facades such as "
                "objc::NSString or objc::handle.");

 public:
  using signature_type = Ret(Args...);

  block() noexcept = default;
  block(std::nullptr_t) noexcept {}

  block(const block &other) : blk_(other.blk_ ? ::_Block_copy(other.blk_) : nullptr) {}

  block(block &&other) noexcept : blk_(std::exchange(other.blk_, nullptr)) {}

  block &operator=(block other) noexcept {
    swap(other);
    return *this;
  }

  ~block() {
    if (blk_) ::_Block_release(blk_);
  }

  void swap(block &other) noexcept { std::swap(blk_, other.blk_); }

  void reset() noexcept {
    if (blk_) ::_Block_release(blk_);
    blk_ = nullptr;
  }

  void *get() const noexcept { return blk_; }
  void *objc_arg() const noexcept { return blk_; }

  explicit operator bool() const noexcept { return blk_ != nullptr; }

  typename detail::block_return_traits<Ret>::cpp_return_type operator()(Args... args) const {
    using traits = detail::block_return_traits<Ret>;
    using RawRet = typename traits::raw_return_type;
    using Fn = RawRet (*)(void *, detail::raw_arg_t<Args>...);
    auto *hdr = static_cast<const detail::block_header *>(blk_);

    if constexpr (std::is_void_v<RawRet>) {
      reinterpret_cast<Fn>(hdr->invoke)(blk_, detail::unwrap_arg(args)...);
    } else {
      RawRet raw = reinterpret_cast<Fn>(hdr->invoke)(blk_, detail::unwrap_arg(args)...);
      return traits::from_raw(raw);
    }
  }

  static block adopt(void *heap) noexcept {
    block b;
    b.blk_ = heap;
    return b;
  }

 private:
  void *blk_ = nullptr;
};

namespace detail {

template <class Ret, class... Args, class F>
block<Ret(Args...)> build_block(F &&f) {
  using return_traits = block_return_traits<Ret>;
  using RawRet = typename return_traits::raw_return_type;

  static_assert(!indirect_return_v<RawRet>,
                "objc::make_block: struct-returned blocks are not supported");

  using Fn = std::decay_t<F>;
  using Lit = block_literal<Fn, Ret, Args...>;

  Lit lit{
      block_header{
          .isa = static_cast<void *>(_NSConcreteStackBlock),
          .flags = block_flags_for<Fn>(),
          .reserved = 0,
          .invoke = reinterpret_cast<void *>(&block_invoke<Lit, Ret, Args...>),
          .descriptor = block_desc<Lit, Fn>(),
      },
      std::forward<F>(f),
  };

  void *heap = _Block_copy(&lit);
  return block<Ret(Args...)>::adopt(heap);
}

template <class R, class... A>
struct callable_sig {
  template <class F>
  static block<R(A...)> make(F &&f) {
    return build_block<R, A...>(std::forward<F>(f));
  }
};

template <class T>
struct fn_sig {
  using type = typename fn_sig<decltype(&T::operator())>::type;
};

template <class R, class... A>
struct fn_sig<R (*)(A...)> {
  using type = callable_sig<R, A...>;
};

template <class R, class... A>
struct fn_sig<R(A...)> {
  using type = callable_sig<R, A...>;
};

template <class C, class R, class... A>
struct fn_sig<R (C::*)(A...)> {
  using type = callable_sig<R, A...>;
};

template <class C, class R, class... A>
struct fn_sig<R (C::*)(A...) const> {
  using type = callable_sig<R, A...>;
};

}  // namespace detail

template <class F>
[[nodiscard]] auto make_block(F &&f) {
  using sig = typename detail::fn_sig<std::remove_cv_t<std::remove_reference_t<F>>>::type;

  return sig::make(std::forward<F>(f));
}

template <class Signature, class F>
[[nodiscard]] block<Signature> make_block(F &&f) {
  using sig = typename detail::fn_sig<Signature>::type;
  return sig::make(std::forward<F>(f));
}

// ===========================================================================
// Example Foundation wrappers
// ===========================================================================

class NSObject : public handle {
 public:
  using handle::handle;

  static Class cls() { return OBJC_CLASS("NSObject"); }
};

class NSString : public NSObject {
 public:
  using NSObject::NSObject;

  static Class cls() { return OBJC_CLASS("NSString"); }

  static object<NSString> from_utf8(const char *s) {
    static const method<autoreleased<NSString>(const char *)> stringWithUTF8String(
        "stringWithUTF8String:");

    return stringWithUTF8String(cls(), s);
  }

  const char *utf8() const OBJC_LIFETIMEBOUND {
    static const method<const char *()> UTF8String("UTF8String");
    return UTF8String(*this);
  }

  std::size_t length() const {
    static const method<std::size_t()> length("length");
    return length(*this);
  }

  object<NSString> copy() const {
    static const method<retained<NSString>()> copy("copy");
    return copy(*this);
  }
};

class NSError : public NSObject {
 public:
  using NSObject::NSObject;

  static Class cls() { return OBJC_CLASS("NSError"); }

  object<NSString> localizedDescription() const {
    static const method<autoreleased<NSString>()> localizedDescription("localizedDescription");
    return localizedDescription(*this);
  }
};

// NSDecimal struct, declared locally so we do not pull in Foundation headers.
// Layout mirrors Foundation/NSDecimal.h.
struct NSDecimal {
  unsigned int meta;
  unsigned short mantissa[8];
};

static_assert(sizeof(NSDecimal) == 20, "NSDecimal must be 20 bytes");

class NSDecimalNumber : public NSObject {
 public:
  using NSObject::NSObject;

  static Class cls() { return OBJC_CLASS("NSDecimalNumber"); }

  static object<NSDecimalNumber> from_string(NSString s) {
    static const method<autoreleased<NSDecimalNumber>(NSString)> decimalNumberWithString(
        "decimalNumberWithString:");

    return decimalNumberWithString(cls(), s);
  }

  NSDecimal value() const {
    static const method<NSDecimal()> decimalValue("decimalValue");
    return decimalValue(*this);
  }
};

class NSAutoreleasePool : public NSObject {
 public:
  using NSObject::NSObject;

  static Class cls() { return OBJC_CLASS("NSAutoreleasePool"); }

  static object<NSAutoreleasePool> create() {
    return alloc_init<NSAutoreleasePool>(OBJC_SEL("init"));
  }
};

}  // namespace objc
