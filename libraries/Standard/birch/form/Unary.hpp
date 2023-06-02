/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

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
  template<class Middle> \
  int rows(const This<Middle>& o) { \
    return rows(eval(o)); \
  } \
  \
  template<class Middle> \
  int columns(const This<Middle>& o) { \
    return columns(eval(o)); \
  } \
  \
  template<class Middle> \
  int length(const This<Middle>& o) { \
    return length(eval(o)); \
  } \
  \
  template<class Middle> \
  int size(const This<Middle>& o) { \
    return size(eval(o)); \
  }

#define BIRCH_UNARY(This, f, ...) \
  template<class Middle> \
  struct is_form<This<Middle>> { \
    static constexpr bool value = true; \
  }; \
  \
  template<class Middle> \
  void reset(This<Middle>& o) { \
    reset(o.m); \
  } \
  \
  template<class Middle> \
  void relink(This<Middle>& o, const RelinkVisitor& visitor) { \
    relink(o.m, visitor); \
  } \
  \
  template<class Middle> \
  void constant(const This<Middle>& o) { \
    constant(o.m); \
  } \
  \
  template<class Middle> \
  bool is_constant(const This<Middle>& o) { \
    return is_constant(o.m); \
  } \
  \
  template<class Middle> \
  void args(This<Middle>& o, const ArgsVisitor& visitor) { \
    args(o.m, visitor); \
  } \
  \
  template<class Middle> \
  void deep_grad(This<Middle>& o, const GradVisitor& visitor) { \
    deep_grad(o.m, visitor); \
  } \
  \
  template<class Middle> \
  auto value(const This<Middle>& o) { \
    return f(value(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Middle> \
  auto eval(const This<Middle>& o) { \
    return f(eval(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Middle> \
  auto peek(const This<Middle>& o) { \
    return f(peek(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Middle> \
  auto move(const This<Middle>& o, const MoveVisitor& visitor) { \
    return f(move(o.m, visitor) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  template<class Middle> \
  auto peg(const This<Middle>& o) { \
    using T = std::decay_t<decltype(peg(o.m))>; \
    return This<T>{peg(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))}; \
  } \
  template<class Middle> \
  auto tag(const This<Middle>& o) { \
    using T = decltype(tag(o.m)); \
    return This<T>{tag(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))}; \
  } \
  \
  template<class Middle> \
  This<Middle>::operator auto() const { \
    return f(value(m) __VA_OPT__(,) __VA_ARGS__); \
  } \
  \
  template<class Middle> \
  auto This<Middle>::operator*() const { \
    return wait(f(value(m) __VA_OPT__(,) __VA_ARGS__)); \
  }

#define BIRCH_UNARY_GRAD(This, f_grad, ...) \
  template<class Middle, class G> \
  void shallow_grad(This<Middle>& o, const G& g, const GradVisitor& visitor) { \
    shallow_grad(o.m, f_grad(g, peek(o), peek(o.m) __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__))), visitor); \
  }

#define BIRCH_UNARY_CONSTRUCT(This, ...) \
  This<decltype(tag(m))>{tag(m) __VA_OPT__(,) __VA_ARGS__}
