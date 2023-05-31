/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_BINARY_FORM(This, f, ...) \
  static constexpr bool is_form = true; \
  Left l; \
  Right r; \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(l, r) \
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
    birch::reset(l); \
    birch::reset(r); \
  } \
  \
  void relink(const RelinkVisitor& visitor) { \
    birch::relink(l, visitor); \
    birch::relink(r, visitor); \
  } \
  \
  void constant() const { \
    birch::constant(l); \
    birch::constant(r); \
  } \
  \
  bool isConstant() const { \
    return birch::is_constant(l) && birch::is_constant(r); \
  } \
  \
  void args(const ArgsVisitor& visitor) { \
    birch::args(l, visitor); \
    birch::args(r, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) { \
    birch::deep_grad(l, visitor); \
    birch::deep_grad(r, visitor); \
  } \
  \
  auto value() const { \
    return f(birch::value(l), birch::value(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return f(birch::eval(l), birch::eval(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return f(birch::peek(l), birch::peek(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return f(birch::move(l, visitor), birch::move(r, visitor)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peg() const { \
    using T = std::decay_t<decltype(birch::peg(l))>; \
    using U = std::decay_t<decltype(birch::peg(r))>; \
    return This<T,U>{birch::peg(l), birch::peg(r)__VA_OPT__(,) __VA_ARGS__}; \
  } \
  \
  auto tag() const { \
    using T = decltype(birch::tag(l)); \
    using U = decltype(birch::tag(r)); \
    return This<T,U>{birch::tag(l), birch::tag(r)__VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_BINARY_GRAD(f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    auto x = birch::peek(*this); \
    auto l1 = birch::peek(l); \
    auto r1 = birch::peek(r); \
    birch::shallow_grad(l, f_grad ## 1(g, x, l1, r1 __VA_OPT__(,) __VA_ARGS__), visitor); \
    birch::shallow_grad(r, f_grad ## 2(g, x, l1, r1 __VA_OPT__(,) __VA_ARGS__), visitor); \
  }

#define BIRCH_BINARY_CONSTRUCT(This, ...) \
  This<decltype(tag(l)),decltype(tag(r))>{tag(l), tag(r) __VA_OPT__(,) __VA_ARGS__}
