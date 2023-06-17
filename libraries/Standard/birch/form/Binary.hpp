/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"
#include "birch/form/Memo.hpp"

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
  template<argument Left, argument Right> \
  int rows(const This<Left,Right>& o) { \
    return rows(eval(o)); \
  } \
  \
  template<argument Left, argument Right> \
  int columns(const This<Left,Right>& o) { \
    return columns(eval(o)); \
  } \
  \
  template<argument Left, argument Right> \
  int length(const This<Left,Right>& o) { \
    return length(eval(o)); \
  } \
  \
  template<argument Left, argument Right> \
  int size(const This<Left,Right>& o) { \
    return size(eval(o)); \
  }

#define BIRCH_BINARY(This, f, ...) \
  template<argument Left, argument Right> \
  struct is_form<This<Left,Right>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Left, argument Right> \
  auto value(const This<Left,Right>& o) { \
    return numbirch::f(value(o.l), value(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Right> \
  auto eval(const This<Left,Right>& o) { \
    return numbirch::f(eval(o.l), eval(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Right> \
  auto peek(const This<Left,Right>& o) { \
    return numbirch::f(peek(o.l), peek(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Right> \
  auto move(const This<Left,Right>& o, const MoveVisitor& visitor) { \
    return numbirch::f(move(o.l, visitor), move(o.r, visitor) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Right> \
  auto peg(const This<Left,Right>& o) { \
    using T = std::decay_t<decltype(peg(o.l))>; \
    using U = std::decay_t<decltype(peg(o.r))>; \
    return This<T,U>{peg(o.l), peg(o.r)}; \
  } \
  \
  template<argument Left, argument Right> \
  void reset(This<Left,Right>& o) { \
    reset(o.l); \
    reset(o.r); \
  } \
  \
  template<argument Left, argument Right> \
  void relink(This<Left,Right>& o, const RelinkVisitor& visitor) { \
    relink(o.l, visitor); \
    relink(o.r, visitor); \
  } \
  \
  template<argument Left, argument Right> \
  void constant(const This<Left,Right>& o) { \
    constant(o.l); \
    constant(o.r); \
  } \
  \
  template<argument Left, argument Right> \
  bool is_constant(const This<Left,Right>& o) { \
    return is_constant(o.l) && is_constant(o.r); \
  } \
  \
  template<argument Left, argument Right> \
  void args(This<Left,Right>& o, const ArgsVisitor& visitor) { \
    args(o.l, visitor); \
    args(o.r, visitor); \
  } \
  \
  template<argument Left, argument Right> \
  void deep_grad(This<Left,Right>& o, const GradVisitor& visitor) { \
    deep_grad(o.l, visitor); \
    deep_grad(o.r, visitor); \
  } \
  \
  template<argument Left, argument Right> \
  This<Left,Right>::operator auto() const { \
    return numbirch::f(value(l), value(r) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  template<argument Left, argument Right> \
  auto This<Left,Right>::operator*() const { \
    return wait(numbirch::f(value(l), value(r) __VA_OPT__(,) __VA_ARGS__)); \
  } \
  \
  \
  template<argument Left, argument Right> \
  auto f(Left&& l, Right&& r __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagLeft = tag_t<Left>; \
    using TagRight = tag_t<Right>; \
    return This<TagLeft,TagRight>{std::forward<Left>(l), \
        std::forward<Right>(r) __VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_BINARY_GRAD(This, f_grad, ...) \
  template<argument Left, argument Right, class G> \
  void shallow_grad(This<Left,Right>& o, const G& g, const GradVisitor& visitor) { \
    auto x = peek(o); \
    auto l1 = peek(o.l); \
    auto r1 = peek(o.r); \
    shallow_grad(o.l, numbirch::f_grad ## 1(g, x, l1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
    shallow_grad(o.r, numbirch::f_grad ## 2(g, x, l1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
  }
