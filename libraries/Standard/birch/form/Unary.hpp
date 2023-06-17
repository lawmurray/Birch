/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"
#include "birch/form/Memo.hpp"

#define BIRCH_UNARY_FORM(This, ...) \
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
  operator auto() const; \
  auto operator*() const;

#define BIRCH_UNARY_SIZE(This, ...) \
  template<argument Middle> \
  int rows(const This<Middle>& o) { \
    return rows(eval(o)); \
  } \
  \
  template<argument Middle> \
  int columns(const This<Middle>& o) { \
    return columns(eval(o)); \
  } \
  \
  template<argument Middle> \
  int length(const This<Middle>& o) { \
    return length(eval(o)); \
  } \
  \
  template<argument Middle> \
  int size(const This<Middle>& o) { \
    return size(eval(o)); \
  }

#define BIRCH_UNARY(This, f, ...) \
  template<argument Middle> \
  struct is_form<This<Middle>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<argument Middle> \
  void reset(This<Middle>& o) { \
    reset(o.m); \
  } \
  \
  template<argument Middle> \
  void relink(This<Middle>& o, const RelinkVisitor& visitor) { \
    relink(o.m, visitor); \
  } \
  \
  template<argument Middle> \
  void constant(const This<Middle>& o) { \
    constant(o.m); \
  } \
  \
  template<argument Middle> \
  bool is_constant(const This<Middle>& o) { \
    return is_constant(o.m); \
  } \
  \
  template<argument Middle> \
  void args(This<Middle>& o, const ArgsVisitor& visitor) { \
    args(o.m, visitor); \
  } \
  \
  template<argument Middle> \
  void deep_grad(This<Middle>& o, const GradVisitor& visitor) { \
    deep_grad(o.m, visitor); \
  } \
  \
  template<argument Middle> \
  auto value(const This<Middle>& o) { \
    return numbirch::f(value(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Middle> \
  auto eval(const This<Middle>& o) { \
    return numbirch::f(eval(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Middle> \
  auto peek(const This<Middle>& o) { \
    return numbirch::f(peek(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Middle> \
  auto move(const This<Middle>& o, const MoveVisitor& visitor) { \
    return numbirch::f(move(o.m, visitor) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<argument Middle> \
  auto peg(const This<Middle>& o) { \
    using T = std::decay_t<decltype(peg(o.m))>; \
    return This<T>{peg(o.m)}; \
  } \
  \
  template<argument Middle> \
  This<Middle>::operator auto() const { \
    return numbirch::f(value(m) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  template<argument Middle> \
  auto This<Middle>::operator*() const { \
    return wait(numbirch::f(value(m) __VA_OPT__(,) __VA_ARGS__)); \
  } \
  \
  template<argument Middle> \
  auto f(Middle&& m __VA_OPT__(, BIRCH_INT(__VA_ARGS__))) { \
    using TagMiddle = tag_t<Middle>; \
    return This<TagMiddle>{std::forward<Middle>(m) \
        __VA_OPT__(,) __VA_ARGS__}; \
  }

#define BIRCH_UNARY_GRAD(This, f_grad, ...) \
  template<argument Middle, class G> \
  void shallow_grad(This<Middle>& o, const G& g, const GradVisitor& visitor) { \
    shallow_grad(o.m, numbirch::f_grad(g, peek(o), peek(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
  }
