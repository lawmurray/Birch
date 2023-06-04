/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_BINARY_FORM(This, ...) \
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
  operator auto() const; \
  auto operator*() const;

#define BIRCH_BINARY_SIZE(This, ...) \
  template<class Left, class Right> \
  int rows(const This<Left,Right>& o) { \
    return rows(eval(o)); \
  } \
  \
  template<class Left, class Right> \
  int columns(const This<Left,Right>& o) { \
    return columns(eval(o)); \
  } \
  \
  template<class Left, class Right> \
  int length(const This<Left,Right>& o) { \
    return length(eval(o)); \
  } \
  \
  template<class Left, class Right> \
  int size(const This<Left,Right>& o) { \
    return size(eval(o)); \
  }

#define BIRCH_BINARY(This, f, ...) \
  template<class Left, class Right> \
  struct is_form<This<Left,Right>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<class Left, class Right> \
  auto value(const This<Left,Right>& o) { \
    return f(value(o.l), value(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Left, class Right> \
  auto eval(const This<Left,Right>& o) { \
    return f(eval(o.l), eval(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Left, class Right> \
  auto peek(const This<Left,Right>& o) { \
    return f(peek(o.l), peek(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Left, class Right> \
  auto move(const This<Left,Right>& o, const MoveVisitor& visitor) { \
    return f(move(o.l, visitor), move(o.r, visitor) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Left, class Right> \
  auto peg(const This<Left,Right>& o) { \
    using T = std::decay_t<decltype(peg(o.l))>; \
    using U = std::decay_t<decltype(peg(o.r))>; \
    return This<T,U>{o}; \
  } \
  \
  template<class Left, class Right> \
  auto tag(const This<Left,Right>& o) { \
    using T = decltype(tag(o.l)); \
    using U = decltype(tag(o.r)); \
    return This<T,U>{o}; \
  } \
  \
  template<class Left, class Right> \
  void reset(This<Left,Right>& o) { \
    reset(o.l); \
    reset(o.r); \
  } \
  \
  template<class Left, class Right> \
  void relink(This<Left,Right>& o, const RelinkVisitor& visitor) { \
    relink(o.l, visitor); \
    relink(o.r, visitor); \
  } \
  \
  template<class Left, class Right> \
  void constant(const This<Left,Right>& o) { \
    constant(o.l); \
    constant(o.r); \
  } \
  \
  template<class Left, class Right> \
  bool is_constant(const This<Left,Right>& o) { \
    return is_constant(o.l) && is_constant(o.r); \
  } \
  \
  template<class Left, class Right> \
  void args(This<Left,Right>& o, const ArgsVisitor& visitor) { \
    args(o.l, visitor); \
    args(o.r, visitor); \
  } \
  \
  template<class Left, class Right> \
  void deep_grad(This<Left,Right>& o, const GradVisitor& visitor) { \
    deep_grad(o.l, visitor); \
    deep_grad(o.r, visitor); \
  } \
  \
  template<class Left, class Right> \
  This<Left,Right>::operator auto() const { \
    return f(value(l), value(r) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  template<class Left, class Right> \
  auto This<Left,Right>::operator*() const { \
    return wait(f(value(l), value(r) __VA_OPT__(,) __VA_ARGS__)); \
  }

#define BIRCH_BINARY_GRAD(This, f_grad, ...) \
  template<class Left, class Right, class G> \
  void shallow_grad(This<Left,Right>& o, const G& g, const GradVisitor& visitor) { \
    auto x = peek(o); \
    auto l1 = peek(o.l); \
    auto r1 = peek(o.r); \
    shallow_grad(o.l, f_grad ## 1(g, x, l1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
    shallow_grad(o.r, f_grad ## 2(g, x, l1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
  }

#define BIRCH_BINARY_CONSTRUCT(This, ...) \
  This<decltype(tag(l)),decltype(tag(r))>{tag(l), tag(r) __VA_OPT__(,) __VA_ARGS__}
