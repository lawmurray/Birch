/**
 * @file
 */
#pragma once

namespace birch {
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
 * Form type.
 */
template<class T>
concept form = is_form_v<T>;

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
 * Expression type.
 */
template<class T>
concept expression = is_expression_v<T>;

/**
 * Argument type.
 * 
 * An argument type is a numeric, form, or expression type.
 */
template<class T>
concept argument = numbirch::numeric<T> || form<T> || expression<T>;

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
  if constexpr (numbirch::future<T>) {
    return x.value();
  } else {
    return std::forward<T>(x);
  }
}

template<argument T>
decltype(auto) value(T&& x) {
  if constexpr (numbirch::numeric<T>) {
    return std::forward<T>(x);
  } else if constexpr (form<T>) {
    return x.value();
  } else if constexpr (expression<T>) {
    return x->value();
  }
}

template<argument T>
decltype(auto) eval(T&& x) {
  if constexpr (numbirch::numeric<T>) {
    return std::forward<T>(x);
  } else if constexpr (form<T>) {
    return x.eval();
  } else if constexpr (expression<T>) {
    return x->eval();
  }
}

template<argument T>
void move(T&& x, const MoveVisitor& visitor) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.move(visitor);
  } else if constexpr (expression<T>) {
    x->move(visitor);
  }
}

template<argument T>
void args(const T& x, const ArgsVisitor& visitor) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.args(visitor);
  } else if constexpr (expression<T>) {
    x->args(visitor);
  }
}

template<argument T>
void reset(T& x) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.reset();
  } else if constexpr (expression<T>) {
    x->reset();
  }
}

template<argument T>
void relink(T& x, const RelinkVisitor& visitor) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.relink(visitor);
  } else if constexpr (expression<T>) {
    x->relink(visitor);
  }
}

template<argument T, numbirch::numeric G>
void grad(const T& x, const G& g) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.grad(g);
  } else if constexpr (expression<T>) {
    x->grad(g);
  }
}

template<argument T, numbirch::numeric G>
void shallow_grad(const T& x, const G& g, const GradVisitor& visitor) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.shallowGrad(g, visitor);
  } else if constexpr (expression<T>) {
    x->shallowGrad(g, visitor);
  }
}

template<argument T>
void deep_grad(const T& x, const GradVisitor& visitor) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.deepGrad(visitor);
  } else if constexpr (expression<T>) {
    x->deepGrad(visitor);
  }
}

template<argument T>
void constant(const T& x) {
  if constexpr (numbirch::numeric<T>) {
    //
  } else if constexpr (form<T>) {
    x.constant();
  } else if constexpr (expression<T>) {
    x->constant();
  }
}

template<argument T>
bool is_constant(const T& x) {
  if constexpr (numbirch::numeric<T>) {
    return true;
  } else if constexpr (form<T>) {
    return x.isConstant();
  } else if constexpr (expression<T>) {
    return x->isConstant();
  }
}

template<argument T>
int rows(const T& x) {
  if constexpr (numbirch::numeric<T>) {
    return numbirch::rows(x);
  } else if constexpr (form<T>) {
    return x.rows();
  } else if constexpr (expression<T>) {
    return x->rows();
  }
}

template<class Arg, class... Args>
int rows(const Arg& arg, const Args&... args) {
  if constexpr (numbirch::is_scalar_v<decltype(eval(arg))>) {
    return rows(args...);
  } else {
    assert((rows(arg) == rows(args...) ||
        numbirch::all_scalar_v<decltype(eval(args))...>) &&
        "incompatible rows");
    return rows(arg);
  }
}

template<argument T>
int columns(const T& x) {
  if constexpr (numbirch::numeric<T>) {
    return numbirch::columns(x);
  } else if constexpr (form<T>) {
    return x.columns();
  } else if constexpr (expression<T>) {
    return x->columns();
  }
}

template<class Arg, class... Args>
int columns(const Arg& arg, const Args&... args) {
  if constexpr (numbirch::is_scalar_v<decltype(eval(arg))>) {
    return columns(args...);
  } else {
    assert((columns(arg) == columns(args...) ||
        numbirch::all_scalar_v<decltype(eval(args))...>) &&
        "incompatible columns");
    return columns(arg);
  }
}

template<argument T>
int length(const T& x) {
  return rows(x);
}

template<argument T>
int size(const T& x) {
  return rows(x)*columns(x);
}

template<class T>
struct tag_s {
  using type = void;
};
template<numbirch::arithmetic T>
struct tag_s<T> {
  using type = std::decay_t<T>;
};
template<numbirch::array T>
struct tag_s<T> {
  using type = std::conditional_t<std::is_rvalue_reference_v<T>,T,
      std::add_lvalue_reference_t<std::add_const_t<T>>>;
};
template<form T>
struct tag_s<T> {
  using type = typename tag_s<std::decay_t<T>>::type;  // specialized for this
};
template<expression T>
struct tag_s<T> {
  using type = std::conditional_t<std::is_rvalue_reference_v<T>,T,
      std::add_lvalue_reference_t<std::add_const_t<T>>>;
};

/**
 * Argument wrapper type for construction of delayed expressions. The
 * following conversions occur:
 * 
 * | Type                 | Conversion           |
 * | -------------------- | -------------------- |
 * | Arithmetic reference | Arithmetic value     |
 * | Array reference      | Array reference      |
 * | Form reference       | Form value           |
 * | Expression reference | Expression reference |
 * 
 * The purpose of `tag_t` is to allow the construction of forms without the
 * overhead of relatively expensive object copies, such as for arrays (which
 * require an allocate and copy, or copy-on-write bookkeeping), or shared
 * pointers (reference counting and atomic operations). Forms can then be
 * rewritten at compile time.
 */
template<argument T>
using tag_t = typename tag_s<T>::type;

template<class T>
struct peg_s {
  using type = void;
};
template<numbirch::arithmetic T>
struct peg_s<T> {
  using type = std::decay_t<T>;
};
template<numbirch::array T>
struct peg_s<T> {
  using type = std::decay_t<T>;
};
template<form T>
struct peg_s<T> {
  using type = typename peg_s<std::decay_t<T>>::type;  // specialized for this
};
template<expression T>
struct peg_s<T> {
  using type = std::decay_t<T>;
};

/**
 * Argument wrapper type for construction of delayed expressions. The
 * following conversions occur:
 * 
 * | Type                 | Conversion       |
 * | -------------------- | ---------------- |
 * | Numeric reference    | Numeric value    |
 * | Form reference       | Form value       |
 * | Expression reference | Expression value |
 * 
 * The purpose of `peg_t` is to solidify a form assembled with tag, and
 * possibly rewritten, turning references into values to ensure that there are
 * no dangling references.
 */
template<argument T>
using peg_t = typename peg_s<T>::type;

/**
 * Argument wrapper function for construction of delayed expressions.
 */
template<argument T>
peg_t<T> peg(T&& x) {
  return peg_t<T>(std::forward<T>(x));
}

/**
 * Argument wrapper function for construction of delayed expressions. The
 * following conversions occur:
 * 
 * | Type                 | Conversion       |
 * | -------------------- | ---------------- |
 * | Numeric reference    | Expression value |
 * | Form reference       | Expression value |
 * | Expression reference | Expression value |
 * 
 * `box()` is used for type erasure of delayed expressions, such as complex
 * forms.
 */
template<argument T>
decltype(auto) box(T&& x) {
  using U = std::decay_t<decltype(wait(eval(x)))>;
  using V = peg_t<T>;
  if constexpr (numbirch::numeric<T>) {
    return Expression<U>(BoxedValue<U>(std::forward<T>(x)));
  } else if constexpr (form<T>) {
    return Expression<U>(BoxedForm<U,V>(V(std::forward<T>(x))));
  } else if constexpr (expression<T>) {
    return std::decay_t<T>(std::forward<T>(x));
  }
}

/**
 * Apply `box()` to multiple arguments and return as a tuple.
 */
template<class... Args>
decltype(auto) box(Args&&... args) {
  return std::make_tuple(box(std::forward<Args>(args))...);
}

/**
 * Argument wrapper function for construction of delayed expressions. The
 * following conversions occur:
 * 
 * | Type                 | Conversion       |
 * | -------------------- | ---------------- |
 * | Numeric reference    | Numeric value    |
 * | Form reference       | Expression value |
 * | Expression reference | Expression value |
 * 
 * `wrap()` is used for partial type erasure of delayed expressions, boxing
 * forms, which can become quite complex, while preserving numeric arguments.
 */
template<class T>
decltype(auto) wrap(T&& x) {
  if constexpr (form<T>) {
    return box(std::forward<T>(x));
  } else {
    return std::decay_t<T>(std::forward<T>(x));
  }
}

/**
 * Apply `wrap()` to multiple arguments and return as a tuple.
 */
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
