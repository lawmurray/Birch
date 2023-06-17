/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"
#include "birch/form/Memo.hpp"

#define BIRCH_TERNARY_FORM(This, ...) \
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
  operator auto() const; \
  auto operator*() const;

#define BIRCH_TERNARY_SIZE(This, ...) \
  template<argument Left, argument Middle, argument Right> \
  int rows(const This<Left,Middle,Right>& o) { \
    return rows(eval(o)); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  int columns(const This<Left,Middle,Right>& o) { \
    return columns(eval(o)); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  int length(const This<Left,Middle,Right>& o) { \
    return length(eval(o)); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  int size(const This<Left,Middle,Right>& o) { \
    return size(eval(o)); \
  }

#define BIRCH_TERNARY(This, f, ...) \
  template<argument Left, argument Middle, argument Right> \
  struct is_form<This<Left,Middle,Right>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Left, argument Middle, argument Right> \
  auto value(const This<Left,Middle,Right>& o) { \
    return numbirch::f(value(o.l), value(o.m), value(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto eval(const This<Left,Middle,Right>& o) { \
    return numbirch::f(eval(o.l), eval(o.m), eval(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto peek(const This<Left,Middle,Right>& o) { \
    return numbirch::f(peek(o.l), peek(o.m), peek(o.r) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto move(const This<Left,Middle,Right>& o, const MoveVisitor& visitor) { \
    return numbirch::f(move(o.l, visitor), move(o.m, visitor), move(o.r, visitor) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto peg(const This<Left,Middle,Right>& o) { \
    using T = std::decay_t<decltype(peg(o.l))>; \
    using U = std::decay_t<decltype(peg(o.m))>; \
    using V = std::decay_t<decltype(peg(o.r))>; \
    return This<T,U,V>{peg(o.l), peg(o.m), peg(o.r)}; \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  void reset(This<Left,Middle,Right>& o) { \
    reset(o.l); \
    reset(o.m); \
    reset(o.r); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  void relink(This<Left,Middle,Right>& o, const RelinkVisitor& visitor) { \
    relink(o.l, visitor); \
    relink(o.m, visitor); \
    relink(o.r, visitor); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  void constant(const This<Left,Middle,Right>& o) { \
    constant(o.l); \
    constant(o.m); \
    constant(o.r); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  bool is_constant(const This<Left,Middle,Right>& o) { \
    return is_constant(o.l) && is_constant(o.m) && is_constant(o.r); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  void args(This<Left,Middle,Right>& o, const ArgsVisitor& visitor) { \
    args(o.l, visitor); \
    args(o.m, visitor); \
    args(o.r, visitor); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  void deep_grad(This<Left,Middle,Right>& o, const GradVisitor& visitor) { \
    deep_grad(o.l, visitor); \
    deep_grad(o.m, visitor); \
    deep_grad(o.r, visitor); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  This<Left,Middle,Right>::operator auto() const { \
    return numbirch::f(value(l), value(m), value(r) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto This<Left,Middle,Right>::operator*() const { \
    return wait(numbirch::f(value(l), value(m), value(r) __VA_OPT__(,) __VA_ARGS__)); \
  } \
  \
  template<argument Left, argument Middle, argument Right> \
  auto f(Left&& l, Middle&& m, Right&& r __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagLeft = tag_t<Left>; \
    using TagMiddle = tag_t<Middle>; \
    using TagRight = tag_t<Right>; \
    return This<TagLeft,TagMiddle,TagRight>{std::forward<Left>(l), \
        std::forward<Middle>(m), std::forward<Right>(r) \
        __VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_TERNARY_GRAD(This, f_grad, ...) \
  template<argument Left, argument Middle, argument Right, class G> \
  void shallow_grad(This<Left,Middle,Right>& o, const G& g, const GradVisitor& visitor) { \
    auto x = peek(o); \
    auto l1 = peek(o.l); \
    auto m1 = peek(o.m); \
    auto r1 = peek(o.r); \
    shallow_grad(o.l, numbirch::f_grad ## 1(g, x, l1, m1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
    shallow_grad(o.m, numbirch::f_grad ## 2(g, x, l1, m1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
    shallow_grad(o.r, numbirch::f_grad ## 3(g, x, l1, m1, r1 __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
  } 
