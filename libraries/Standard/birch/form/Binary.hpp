/**
 * @file
 */
#pragma once

#include "birch/form/Memo.hpp"

#define BIRCH_BINARY_FORM(This, ...) \
  Left l; \
  Right r; \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(l, r) \
  \
  This(const This&) = default; \
  This(This&&) = default; \
  \
  template<argument T1, argument U1> \
  This(T1&& l, U1&& r __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) : \
      l(std::forward<T1>(l)), \
      r(std::forward<U1>(r)) \
      __VA_OPT__(, BIRCH_INIT(__VA_ARGS__)) {} \
  \
  template<argument T1, argument U1> \
  This(const This<T1,U1>& o) : \
      l(o.l), \
      r(o.r) \
      __VA_OPT__(, BIRCH_COPY_INIT(__VA_ARGS__)) {} \
  \
  template<argument T1, argument U1> \
  This(This<T1,U1>&& o) : \
     l(std::forward<decltype(o.l)>(o.l)), \
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
  void args(const ArgsVisitor& visitor) const { \
    birch::args(l, visitor); \
    birch::args(r, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) const { \
    birch::deep_grad(l, visitor); \
    birch::deep_grad(r, visitor); \
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

#define BIRCH_BINARY_SIZE(This, ...) \
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

#define BIRCH_BINARY_EVAL(This, f, ...) \
  auto eval() const { \
    return numbirch::f(birch::eval(l), birch::eval(r) \
        __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  auto value() const { \
    constant(); \
    return eval(); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return numbirch::f(birch::move(l, visitor), birch::move(r, visitor) \
        __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_BINARY_GRAD(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    auto l1 = birch::eval(l); \
    auto r1 = birch::eval(r); \
    if (!is_constant(l)) { \
      birch::shallow_grad(l, numbirch::f_grad ## 1(g, l1, r1 \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
    if (!is_constant(r)) { \
      birch::shallow_grad(r, numbirch::f_grad ## 2(g, l1, r1 \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
  }

#define BIRCH_BINARY_GRAD_WITH_RESULT(This, f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) const { \
    auto x = eval(); \
    auto l1 = birch::eval(l); \
    auto r1 = birch::eval(r); \
    if (!is_constant(l)) { \
      birch::shallow_grad(l, numbirch::f_grad ## 1(g, x, l1, r1 \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
    if (!is_constant(r)) { \
      birch::shallow_grad(r, numbirch::f_grad ## 2(g, x, l1, r1 \
          __VA_OPT__(,) __VA_ARGS__), visitor); \
    } \
  }

#define BIRCH_BINARY_CALL(This, f, ...) \
  template<class Left, class Right> \
  auto f(Left&& l, Right&& r __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagLeft = tag_t<Left>; \
    using TagRight = tag_t<Right>; \
    return This<TagLeft,TagRight>(std::forward<Left>(l), \
        std::forward<Right>(r) __VA_OPT__(,) __VA_ARGS__); \
  }

#define BIRCH_BINARY_TYPE(This, ...) \
  template<argument Left, argument Right> \
  struct is_form<This<Left,Right>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Left, argument Right> \
  struct tag_s<This<Left,Right>> { \
    using type = This<tag_t<Left>,tag_t<Right>>; \
  }; \
  \
  template<argument Left, argument Right> \
  struct peg_s<This<Left,Right>> { \
    using type = This<peg_t<Left>,peg_t<Right>>; \
  };
