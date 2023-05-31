/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

#define BIRCH_NULLARY_FORM(This, f, ...) \
  static constexpr bool is_form = true; \
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
  operator auto() const { \
    return f(__VA_ARGS__); \
  } \
  \
  decltype(wait(f(__VA_ARGS__))) operator*() const { \
    return wait(f(__VA_ARGS__)); \
  } \
  \
  static constexpr void reset() {} \
  static constexpr void relink(const RelinkVisitor& visitor) {} \
  static constexpr void constant() {} \
  \
  bool isConstant() const { \
    return true; \
  } \
  \
  static constexpr void args(const ArgsVisitor& visitor) {} \
  static constexpr void deepGrad(const GradVisitor& visitor) {} \
  \
  auto value() const { \
    return f(__VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return f(__VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return f(__VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return f(__VA_ARGS__); \
  } \
  \
  auto peg() const { \
    return This{__VA_ARGS__}; \
  } \
  \
  auto tag() const { \
    return This{__VA_ARGS__}; \
  }

#define BIRCH_NULLARY_GRAD(f_grad, ...) \
  template<class G> \
  static constexpr void shallowGrad(const G& g, const GradVisitor& visitor) {}

#define BIRCH_NULLARY_CONSTRUCT(This, ...) \
  This{__VA_ARGS__}
