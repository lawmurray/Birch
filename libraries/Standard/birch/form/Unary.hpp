/**
 * @file
 */
#pragma once

#include "birch/form/Memo.hpp"

#define BIRCH_UNARY_FORM(This, ...) \
  Middle m; \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(m) \
  \
  This(const This&) = default;  \
  This(This&&) = default; \
  \
  This(const Middle& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) : \
      m(m) __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  This(Middle& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) \
      requires (!std::same_as<const Middle&,Middle&>) : \
      m(m) __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  This(Middle&& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) \
      requires (!std::same_as<const Middle&,Middle&&>) : \
      m(std::move(m)) __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  template<argument O> \
  This(const O& o) : \
      m(o.m) __VA_OPT__(, BIRCH_COPY_INIT(__VA_ARGS__)) {} \
  \
  template<argument O> \
  This(O&& o) : \
     m(std::move(o.m)) __VA_OPT__(, BIRCH_MOVE_INIT(__VA_ARGS__)) {} \
  \
  auto operator->() { \
    return this; \
  } \
  \
  auto operator->() const { \
    return this; \
  } \
  \
  void reset() { \
    birch::reset(m); \
  } \
  \
  void relink(const RelinkVisitor& visitor) { \
    birch::relink(m, visitor); \
  } \
  \
  void constant() const { \
    birch::constant(m); \
  } \
  \
  bool isConstant() const { \
    return birch::is_constant(m); \
  } \
  \
  void args(const ArgsVisitor& visitor) const { \
    birch::args(m, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) const { \
    birch::deep_grad(m, visitor); \
  }

#define BIRCH_UNARY_SIZE(This, ...) \
  int rows() const { \
    return birch::rows(eval()); \
  } \
  \
  int columns() const { \
    return birch::columns(eval()); \
  } \
  \
  int length() const { \
    return birch::length(eval()); \
  } \
  \
  int size() const { \
    return birch::size(eval()); \
  }

#define BIRCH_UNARY_EVAL(This, f, ...) \
  auto value() const { \
    return numbirch::f(birch::value(m) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return numbirch::f(birch::eval(m) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return numbirch::f(birch::peek(m) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return numbirch::f(birch::move(m, visitor) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  operator auto() const { \
    return value(); \
  } \
  \
  auto operator*() const { \
    return wait(value()); \
  }

#define BIRCH_UNARY_GRAD(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    birch::shallow_grad(m, numbirch::f_grad(g, peek(), birch::peek(m) \
        __VA_OPT__(,) __VA_ARGS__), visitor); \
  }

#define BIRCH_UNARY_CALL(This, f, ...) \
  template<argument Middle> \
  auto f(Middle&& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagMiddle = tag_t<Middle>; \
    return This<TagMiddle>(std::forward<Middle>(m) \
        __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_UNARY_TYPE(This, ...) \
  template<argument Middle> \
  struct is_form<This<Middle>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Middle> \
  struct peg_s<This<Middle>> { \
    using type = This<peg_t<Middle>>; \
  };
