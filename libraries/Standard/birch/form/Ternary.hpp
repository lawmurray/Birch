/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_TERNARY_FORM(This, f, ...) \
  static constexpr bool is_form = true; \
  Left l; \
  Middle m; \
  Right r; \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(l, m, r) \
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
    birch::reset(m); \
    birch::reset(r); \
  } \
  \
  void relink(const RelinkVisitor& visitor) { \
    birch::relink(l, visitor); \
    birch::relink(m, visitor); \
    birch::relink(r, visitor); \
  } \
  \
  void constant() const { \
    birch::constant(l); \
    birch::constant(m); \
    birch::constant(r); \
  } \
  \
  bool isConstant() const { \
    return birch::is_constant(l) && birch::is_constant(m) && \
        birch::is_constant(r); \
  } \
  \
  void args(const ArgsVisitor& visitor) { \
    birch::args(l, visitor); \
    birch::args(m, visitor); \
    birch::args(r, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) { \
    birch::deep_grad(l, visitor); \
    birch::deep_grad(m, visitor); \
    birch::deep_grad(r, visitor); \
  } \
  \
  auto value() const { \
    return f(birch::value(l), birch::value(m), birch::value(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return f(birch::eval(l), birch::eval(m), birch::eval(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return f(birch::peek(l), birch::peek(m), birch::peek(r)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return f(birch::move(l, visitor), birch::move(m, visitor), birch::move(r, visitor)__VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peg() const { \
    using T = std::decay_t<decltype(birch::peg(l))>; \
    using U = std::decay_t<decltype(birch::peg(m))>; \
    using V = std::decay_t<decltype(birch::peg(r))>; \
    return This<T,U,V>{birch::peg(l), birch::peg(m), birch::peg(r)__VA_OPT__(,) __VA_ARGS__}; \
  } \
  auto tag() const { \
    using T = decltype(birch::tag(l)); \
    using U = decltype(birch::tag(m)); \
    using V = decltype(birch::tag(r)); \
    return This<T,U,V>{birch::tag(l), birch::tag(m), birch::tag(r)__VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_TERNARY_GRAD(f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    auto x = birch::peek(*this); \
    auto l1 = birch::peek(l); \
    auto m1 = birch::peek(m); \
    auto r1 = birch::peek(r); \
    birch::shallow_grad(l, f_grad ## 1(g, x, l1, m1, r1 __VA_OPT__(,) __VA_ARGS__), visitor); \
    birch::shallow_grad(m, f_grad ## 2(g, x, l1, m1, r1 __VA_OPT__(,) __VA_ARGS__), visitor); \
    birch::shallow_grad(r, f_grad ## 3(g, x, l1, m1, r1 __VA_OPT__(,) __VA_ARGS__), visitor); \
  } 

#define BIRCH_TERNARY_CONSTRUCT(This, ...) \
  This<decltype(tag(l)),decltype(tag(m)),decltype(tag(r))>{tag(l), tag(m), tag(r) __VA_OPT__(,) __VA_ARGS__}
