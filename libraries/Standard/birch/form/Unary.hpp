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
  template<argument T1> \
  This(std::in_place_t, T1&& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) : \
      m(std::forward<T1>(m)) \
      __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  template<argument... Args> /* parameter pack to support Cast<To,Middle> */ \
  This(const This<Args...>& o) : \
      m(o.m) \
      __VA_OPT__(, BIRCH_COPY_INIT(__VA_ARGS__)) {} \
  \
  template<argument... Args> /* parameter pack to support Cast<To,Middle> */ \
  This(This<Args...>&& o) : \
      m(std::forward<decltype(o.m)>(o.m)) \
     __VA_OPT__(, BIRCH_MOVE_INIT(__VA_ARGS__)) {} \
  \
  auto operator->() { \
    return this; \
  } \
  \
  auto operator->() const { \
    return this; \
  } \
  \
  operator auto() const { \
    return value(); \
  } \
  \
  auto operator*() const { \
    return wait(value()); \
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
  } \
  \
  template<class Buffer> \
  void write(const Buffer& buffer) const { \
    buffer->set(value()); \
  } \
  \
  template<class Buffer> \
  void write(const Integer t, const Buffer& buffer) const { \
    buffer->set(value()); \
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
  auto move(const MoveVisitor& visitor) const { \
    return numbirch::f(birch::move(m, visitor) __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_UNARY_GRAD_WITH_RESULT(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    if (!is_constant(m)) { \
      birch::shallow_grad(m, numbirch::f_grad(g, eval(), birch::eval(m) \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
  }

#define BIRCH_UNARY_GRAD(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    if (!is_constant(m)) { \
      birch::shallow_grad(m, numbirch::f_grad(g, birch::eval(m) \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
  }

#define BIRCH_UNARY_CALL(This, f, ...) \
  template<argument Middle> \
  auto f(Middle&& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagMiddle = tag_t<Middle>; \
    return This<TagMiddle>(std::in_place, std::forward<Middle>(m) \
        __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_UNARY_TYPE(This, ...) \
  template<argument Middle> \
  struct is_form<This<Middle>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Middle> \
  struct tag_s<This<Middle>> { \
    using type = This<tag_t<Middle>>; \
  }; \
  \
  template<argument Middle> \
  struct peg_s<This<Middle>> { \
    using type = This<peg_t<Middle>>; \
  };
