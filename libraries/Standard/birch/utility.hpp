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
private:
  template<class U>
  static constexpr bool test(decltype(&U::is_form)) {
    return true;
  }
  template<class>
  static constexpr bool test(...) {
    return false;
  }

public:
  static constexpr bool value = test<T>(0);
};
template<class T>
inline constexpr bool is_form_v = is_form<std::decay_t<T>>::value;

/**
 * Is `T` an expression type?
 */
template<class T>
struct is_expression {
private:
  template<class U>
  static constexpr bool test(
        typename std::decay_t<U>::value_type::Value_*) {
    return std::is_base_of_v<
        Expression_<typename std::decay_t<U>::value_type::Value_>,
        typename std::decay_t<U>::value_type>;
  }
  template<class>
  static constexpr bool test(...) {
    return false;
  }

public:
  static constexpr bool value = test<T>(0);
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
 * Are all of the given types either of arithmetic, array, Form, or Expression
 * type?
 */
template<class Arg, class... Args>
struct all_numerical {
  static constexpr bool value = is_numerical_v<Arg> &&
      all_numerical<Args...>::value;
};
template<class Arg>
struct all_numerical<Arg> {
  static constexpr bool value = is_numerical_v<Arg>;
};
template<class... Args>
inline constexpr bool all_numerical_v = all_numerical<Args...>::value;

/**
 * Are any of the given types of a form or expression type?
 */
template<class Arg, class... Args>
struct any_form_or_expression {
  static constexpr bool value = is_form_v<Arg> || is_expression_v<Arg> ||
      any_form_or_expression<Args...>::value;
};
template<class Arg>
struct any_form_or_expression<Arg> {
  static constexpr bool value = is_form_v<Arg> || is_expression_v<Arg>;
};
template<class... Args>
inline constexpr bool any_form_or_expression_v =
    any_form_or_expression<Args...>::value;

/**
 * Are the given types compatible as operands in a delayed expression?
 */
template<class... Args>
struct is_delay {
  static constexpr bool value = all_numerical_v<Args...> &&
      any_form_or_expression_v<Args...>;
};
template<class... Args>
inline constexpr bool is_delay_v = is_delay<Args...>::value;

/**
 * Construct an object.
 *
 * @tparam Type The class type.
 * @tparam Args Argument types.
 *
 * @return The object.
 */
template<class Type, class... Args>
Type construct(Args&&... args) {
  return Type(std::forward<Args>(args)...);
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

template<class T>
auto rows(const T& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return numbirch::rows(x);
  } else if constexpr (is_expression_v<T>) {
    return x->rows();
  } else {
    return x.rows();
  }
}

template<class T>
auto columns(const T& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return numbirch::columns(x);
  } else if constexpr (is_expression_v<T>) {
    return x->columns();
  } else {
    return x.columns();
  }
}

template<class T>
auto length(const T& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return numbirch::length(x);
  } else if constexpr (is_expression_v<T>) {
    return x->length();
  } else {
    return x.length();
  }
}

template<class T>
auto size(const T& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return numbirch::size(x);
  } else if constexpr (is_expression_v<T>) {
    return x->size();
  } else {
    return x.size();
  }
}

template<class T>
decltype(auto) value(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return std::forward<T>(x);
  } else if constexpr (is_expression_v<T>) {
    return x->value();
  } else {
    return x.value();
  }
}

template<class T>
decltype(auto) eval(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return std::forward<T>(x);
  } else if constexpr (is_expression_v<T>) {
    return x->eval();
  } else {
    return x.eval();
  }
}

template<class T>
decltype(auto) peek(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return std::forward<T>(x);
  } else if constexpr (is_expression_v<T>) {
    return x->peek();
  } else {
    return x.peek();
  }
}

template<class T>
decltype(auto) move(T&& x, const MoveVisitor& visitor) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return std::forward<T>(x);
  } else if constexpr (is_expression_v<T>) {
    return x->move(visitor);
  } else {
    return x.move(visitor);
  }
}

template<class T>
void args(T&& x, const ArgsVisitor& visitor) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->args(visitor);
  } else {
    x.args(visitor);
  }
}

template<class T>
void reset(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->reset();
  } else {
    x.reset();
  }
}

template<class T>
void relink(T&& x, const RelinkVisitor& visitor) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->relink(visitor);
  } else {
    x.relink(visitor);
  }
}

template<class T, class G>
void grad(T&& x, const G& g) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->grad(g);
  } else {
    x.grad(g);
  }
}

template<class T, class G>
void shallow_grad(T&& x, const G& g, const GradVisitor& visitor) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->shallowGrad(g, visitor);
  } else {
    x.shallowGrad(g, visitor);
  }
}

template<class T>
void deep_grad(T&& x, const GradVisitor& visitor) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->deepGrad(visitor);
  } else {
    x.deepGrad(visitor);
  }
}

template<class T>
void constant(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    //
  } else if constexpr (is_expression_v<T>) {
    x->constant();
  } else {
    x.constant();
  }
}

template<class T>
bool is_constant(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return true;
  } else if constexpr (is_expression_v<T>) {
    return x->isConstant();
  } else {
    return x.isConstant();
  }
}

template<class T>
decltype(auto) peg(const T& x) {
  if constexpr (is_form_v<T>) {
    return x.peg();
  } else {
    return std::remove_const_t<std::remove_reference_t<T>>(x);
  }
}

template<class T>
decltype(auto) tag(const T& x) {
  if constexpr (is_form_v<T>) {
    return x.tag();
  } else if constexpr (numbirch::is_arithmetic_v<T>) {
    return std::remove_const_t<std::remove_reference_t<T>>(x);
  } else {
    return x;
  }
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
    return std::forward<T>(x);
  }
}

template<class... Args>
decltype(auto) box(Args&&... args) {
  return std::make_tuple(box(std::forward<Args>(args))...);
}

template<class T>
decltype(auto) wrap(T&& x) {
  if constexpr (numbirch::is_numeric_v<T>) {
    return x;
  } else if constexpr (is_form_v<T>) {
    return box(x);
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
