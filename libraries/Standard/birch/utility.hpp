/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Is `T` an array type?
 */
template<class T>
struct is_array {
  static constexpr bool value = false;
};
template<class T, int D>
struct is_array<numbirch::Array<T,D>> {
  static constexpr bool value = true;
};
template<class T>
inline constexpr bool is_array_v = is_array<std::decay_t<T>>::value;

/**
 * Is `T` a future type?
 */
template<class T>
struct is_future {
  static constexpr bool value = false;
};
template<class T>
struct is_future<numbirch::Array<T,0>> {
  static constexpr bool value = true;
};
template<class T>
inline constexpr bool is_future_v = is_future<std::decay_t<T>>::value;

/**
 * Is `T` a form type?
 */
template<class T>
struct is_form {
  static constexpr bool value = false;
};
template<class T>
inline constexpr bool is_form_v = is_form<std::decay_t<T>>::value;

/**
 * Is `T` an expression type?
 */
template<class T>
struct is_expression {
  static constexpr bool value = false;
};
template<class T>
inline constexpr bool is_expression_v = is_expression<std::decay_t<T>>::value;

/**
 * Is `T` either of arithmetic, array, form, or expression type?
 */
template<class T>
struct is_numerical {
  static constexpr bool value = numbirch::is_numeric_v<T> || is_form_v<T> ||
      is_expression_v<T>;
};
template<class T>
inline constexpr bool is_numerical_v = is_numerical<std::decay_t<T>>::value;

/**
 * Make a shared object.
 *
 * @tparam T Type.
 * @tparam Args Argument types.
 * 
 * @param args Arguments.
 *
 * @return If the type is constructible with the given arguments, then an
 * optional with a so-constructed value, otherwise an optional with no value.
 */
template<class T, class... Args, std::enable_if_t<
    membirch::is_pointer<T>::value &&
    std::is_constructible_v<typename T::value_type,Args...>,int> = 0>
std::optional<T> make_optional(Args&&... args) {
  return T(std::forward<Args>(args)...);
}

/**
 * Make a shared object.
 *
 * @tparam T Type.
 * @tparam Args Argument types.
 * 
 * @param args Arguments.
 *
 * @return If the type is constructible with the given arguments, then an
 * optional with a so-constructed value, otherwise an optional with no value.
 */
template<class T, class... Args, std::enable_if_t<
    membirch::is_pointer<T>::value &&
    !std::is_constructible_v<typename T::value_type,Args...>,int> = 0>
std::optional<T> make_optional(Args&&... args) {
  return std::nullopt;
}

/**
 * Make a value or object.
 *
 * @tparam T Type.
 * @tparam Args Argument types.
 * 
 * @param args Arguments.
 *
 * @return If the type is constructible with the given arguments, then an
 * optional with a so-constructed value, otherwise an optional with no value.
 */
template<class T, class... Args, std::enable_if_t<
    !membirch::is_pointer<T>::value &&
    std::is_constructible_v<T,Args...>,int> = 0>
std::optional<T> make_optional(Args&&... args) {
  return T(std::forward<Args>(args)...);
}

/**
 * Make a value or object.
 *
 * @tparam T Type.
 * @tparam Args Argument types.
 * 
 * @param args Arguments.
 *
 * @return If the type is constructible with the given arguments, then an
 * optional with a so-constructed value, otherwise an optional with no value.
 */
template<class T, class... Args, std::enable_if_t<
    !membirch::is_pointer<T>::value &&
    !std::is_constructible_v<T,Args...>,int> = 0>
std::optional<T> make_optional(Args&&... args) {
  return std::nullopt;
}

/**
 * Cast of anything to itelf.
 */
template<class To, class From,
    std::enable_if_t<std::is_same_v<To,From>,int> = 0>
std::optional<To> optional_cast(const From& from) {
  return from;
}

/**
 * Cast of a pointer.
 */
template<class To, class From,
    std::enable_if_t<!std::is_same_v<To,From> &&
    membirch::is_pointer<To>::value && membirch::is_pointer<From>::value,int> = 0>
std::optional<To> optional_cast(const From& from) {
  auto ptr = dynamic_cast<typename To::value_type*>(from.get());
  if (ptr) {
    return To(ptr);
  } else {
    return std::nullopt;
  }
}

/**
 * Cast of a non-pointer.
 */
template<class To, class From,
    std::enable_if_t<!std::is_same_v<To,From> &&
    std::is_constructible_v<To,From> &&
    (!membirch::is_pointer<To>::value ||
    !membirch::is_pointer<From>::value),int> = 0>
std::optional<To> optional_cast(const From& from) {
  return To(from);
}

/**
 * Non-identity cast of a non-pointer.
 */
template<class To, class From,
    std::enable_if_t<!std::is_same_v<To,From> &&
    !std::is_constructible_v<To,From> &&
    (!membirch::is_pointer<To>::value ||
    !membirch::is_pointer<From>::value),int> = 0>
std::optional<To> optional_cast(const From& from) {
  return std::nullopt;
}

/**
 * Cast of an optional of anything.
 */
template<class To, class From>
std::optional<To> optional_cast(const std::optional<From>& from) {
  if (from.has_value()) {
    return optional_cast<To>(from.value());
  } else {
    return std::nullopt;
  }
}

template<class T>
decltype(auto) wait(T&& x) {
  if constexpr (is_future_v<T>) {
    return x.value();
  } else {
    return std::forward<T>(x);
  }
}

using numbirch::rows;
using numbirch::columns;
using numbirch::length;
using numbirch::size;

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr decltype(auto) value(T&& x) {
  return std::forward<T>(x);
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr decltype(auto) eval(T&& x) {
  return std::forward<T>(x);
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr decltype(auto) peek(T&& x) {
  return std::forward<T>(x);
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr decltype(auto) move(T&& x, const MoveVisitor& visitor) {
  return std::forward<T>(x);
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void args(const T& x, const ArgsVisitor& visitor) {
  //
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void reset(T& x) {
  //
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void relink(T& x, const RelinkVisitor& visitor) {
  //
}

template<class T, class G, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void grad(T& x, const G& g) {
  //
}

template<class T, class G, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void shallow_grad(T& x, const G& g, const GradVisitor& visitor) {
  //
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void deep_grad(T& x, const GradVisitor& visitor) {
  //
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr void constant(const T& x) {
  //
}

template<class T, std::enable_if_t<numbirch::is_numeric_v<T>,int> = 0>
constexpr bool is_constant(const T& x) {
  return true;
}

template<class T>
constexpr decltype(auto) peg(const T& x) {
  return std::remove_const_t<std::remove_reference_t<T>>(x);
}

template<class T>
constexpr decltype(auto) tag(const T& x) {
  return std::remove_const_t<std::remove_reference_t<T>>(x);
}

template<class T>
decltype(auto) box(T&& x) {
  using U = typename std::decay_t<decltype(wait(eval(x)))>;
  using V = typename std::decay_t<decltype(peg(x))>;
  if constexpr (numbirch::is_numeric_v<T>) {
    return Expression<U>(BoxedValue<U>(std::forward<T>(x)));
  } else if constexpr (is_form_v<T>) {
    return Expression<U>(BoxedForm<U,V>(peg(std::forward<T>(x))));
  } else {
    return Expression<U>(std::forward<T>(x));
  }
}

template<class... Args>
decltype(auto) box(Args&&... args) {
  return std::make_tuple(box(std::forward<Args>(args))...);
}

template<class T>
decltype(auto) wrap(T&& x) {
  if constexpr (is_form_v<T>) {
    return box(std::forward<T>(x));
  } else {
    return std::forward<T>(x);
  }
}

template<class... Args>
decltype(auto) wrap(Args&&... args) {
  return std::make_tuple(wrap(std::forward<Args>(args))...);
}

/**
 * Optional assign. Corresponds to the `<-?` operator in Birch.
 *
 * @param to Target.
 * @param from Source.
 *
 * If @p from has a value, then assign it to @p to, otherwise do nothing.
 */
template<class To, class From>
To& optional_assign(To& to, const std::optional<From>& from) {
  if (from.has_value()) {
    to = from.value();
  }
  return to;
}

/**
 * Optional assign to an object. Corresponds to the `<-?` operator in Birch.
 *
 * @param to Target.
 * @param from Source.
 *
 * If @p from has a value, then assign it to @p to, otherwise do nothing.
 */
template<class To, class From>
const membirch::Shared<To>& optional_assign(const membirch::Shared<To>& to,
    const std::optional<From>& from) {
  if (from.has_value()) {
    to = from.value();
  }
  return to;
}

}
