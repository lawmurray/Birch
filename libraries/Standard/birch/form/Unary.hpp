/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_UNARY_FORM(This, f, ...) \
  static constexpr bool is_form = true; \
  Middle m; \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(m) \
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
  void args(const ArgsVisitor& visitor) { \
    birch::args(m, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) { \
    birch::deep_grad(m, visitor); \
  } \
  \
  auto value() const { \
    return f(birch::value(m)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return f(birch::eval(m)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return f(birch::peek(m)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return f(birch::move(m, visitor)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peg() const { \
    using T = std::decay_t<decltype(birch::peg(m))>; \
    return This<T>{birch::peg(m)__VA_OPT__(,) __VA_ARGS__}; \
  } \
  auto tag() const { \
    using T = decltype(birch::tag(m)); \
    return This<T>{birch::tag(m)__VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_UNARY_GRAD(f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    birch::shallow_grad(m, f_grad(g, birch::peek(*this), birch::peek(m)__VA_OPT__(,) __VA_ARGS__), visitor); \
  }

#define BIRCH_UNARY_CONSTRUCT(This, ...) \
  This<decltype(tag(m))>{tag(m) __VA_OPT__(,) __VA_ARGS__}
