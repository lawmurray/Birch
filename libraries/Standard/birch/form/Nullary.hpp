/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_NULLARY_FORM(This, ...) \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This, void) \
  MEMBIRCH_STRUCT_MEMBERS() \
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
  auto operator*() const; \

#define BIRCH_NULLARY(This, f, ...) \
  template<> \
  struct is_form<This> { \
    static constexpr bool value = true; \
  }; \
  \
  inline auto value(const This& o) { \
    return f(__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  inline auto eval(const This& o) { \
    return f(__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  inline auto peek(const This& o) { \
    return f(__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  inline auto move(const This& o, const MoveVisitor& visitor) { \
    return f(__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))); \
  } \
  \
  inline auto peg(const This& o) { \
    return This{__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))}; \
  } \
  \
  inline auto tag(const This& o) { \
    return This{__VA_OPT__(BIRCH_O_DOT(__VA_ARGS__))}; \
  } \
  \
  static constexpr void reset(const This&) {} \
  static constexpr void relink(const This&, const RelinkVisitor&) {} \
  static constexpr void constant(const This&) {} \
  static constexpr bool is_constant(const This&) { return true; } \
  static constexpr void args(const This&, const ArgsVisitor&) {} \
  static constexpr void deep_grad(const This&, const GradVisitor&) {} \
  \
  inline This::operator auto() const { \
    return f(__VA_ARGS__); \
  } \
  \
  inline auto This::operator*() const { \
    return wait(f(__VA_ARGS__)); \
  }

#define BIRCH_NULLARY_SIZE(This, ...) \
  inline int rows(const This& o) { \
    return rows(eval(o)); \
  } \
  \
  inline int columns(const This& o) { \
    return columns(eval(o)); \
  } \
  \
  inline int length(const This& o) { \
    return length(eval(o)); \
  } \
  \
  inline int size(const This& o) { \
    return size(eval(o)); \
  }

#define BIRCH_NULLARY_GRAD(This, f_grad, ...) \
  template<class G> \
  static constexpr void shallow_grad(const This&, const G&, __VA_OPT__(, BIRCH_O_DOT(__VA_ARGS__)), const GradVisitor&) {}

#define BIRCH_NULLARY_CONSTRUCT(This, ...) \
  This{__VA_ARGS__}
