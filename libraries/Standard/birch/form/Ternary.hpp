/**
 * @file
 */
#pragma once

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
  This(const This&) = default; \
  This(This&&) = default; \
  \
  template<argument T1, argument U1, argument V1> \
  This(T1&& l, U1&& m, V1&& r __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) : \
      l(std::forward<T1>(l)), \
      m(std::forward<U1>(m)), \
      r(std::forward<V1>(r)) \
      __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  template<argument T1, argument U1, argument V1> \
  This(const This<T1,U1,V1>& o) : \
      l(o.l), \
      m(o.m), \
      r(o.r) \
      __VA_OPT__(, BIRCH_COPY_INIT(__VA_ARGS__)) {} \
  \
  template<argument T1, argument U1, argument V1> \
  This(This<T1,U1,V1>&& o) : \
     l(std::forward<decltype(o.l)>(o.l)), \
     m(std::forward<decltype(o.m)>(o.m)), \
     r(std::forward<decltype(o.r)>(o.r)) \
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
    return is_constant(l) && is_constant(m) && is_constant(r); \
  } \
  \
  void args(const ArgsVisitor& visitor) const { \
    birch::args(l, visitor); \
    birch::args(m, visitor); \
    birch::args(r, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) const { \
    birch::deep_grad(l, visitor); \
    birch::deep_grad(m, visitor); \
    birch::deep_grad(r, visitor); \
  }

#define BIRCH_TERNARY_SIZE(This, ...) \
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

#define BIRCH_TERNARY_EVAL(This, f, ...) \
  auto value() const { \
    return numbirch::f(birch::value(l), birch::value(m), birch::value(r) \
       __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return numbirch::f(birch::eval(l), birch::eval(m), birch::eval(r) \
        __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return numbirch::f(birch::peek(l), birch::peek(m), birch::peek(r) \
        __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return numbirch::f(birch::move(l, visitor), birch::move(m, visitor), \
        birch::move(r, visitor) __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_TERNARY_GRAD(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    auto x = peek(); \
    auto l1 = birch::peek(l); \
    auto m1 = birch::peek(m); \
    auto r1 = birch::peek(r); \
    birch::shallow_grad(l, numbirch::f_grad ## 1(g, x, l1, m1, r1 \
        __VA_OPT__(,) __VA_ARGS__), visitor); \
    birch::shallow_grad(m, numbirch::f_grad ## 2(g, x, l1, m1, r1 \
        __VA_OPT__(,) __VA_ARGS__), visitor); \
    birch::shallow_grad(r, numbirch::f_grad ## 3(g, x, l1, m1, r1 \
        __VA_OPT__(,) __VA_ARGS__), visitor); \
  } 

#define BIRCH_TERNARY_CALL(This, f, ...) \
  template<argument Left, argument Middle, argument Right> \
  auto f(Left&& l, Middle&& m, Right&& r \
      __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagLeft = tag_t<Left>; \
    using TagMiddle = tag_t<Middle>; \
    using TagRight = tag_t<Right>; \
    return This<TagLeft,TagMiddle,TagRight>{std::forward<Left>(l), \
        std::forward<Middle>(m), std::forward<Right>(r) \
        __VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_TERNARY_TYPE(This, ...) \
  template<argument Left, argument Middle, argument Right> \
  struct is_form<This<Left,Middle,Right>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Left, argument Middle, argument Right> \
  struct tag_s<This<Left,Middle,Right>> { \
    using type = This<tag_t<Left>,tag_t<Middle>,tag_t<Right>>; \
  }; \
  \
  template<argument Left, argument Middle, argument Right> \
  struct peg_s<This<Left,Middle,Right>> { \
    using type = This<peg_t<Left>,peg_t<Middle>,peg_t<Right>>; \
  };
