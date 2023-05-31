/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"
#include "birch/form/Unary.hpp"

#define BIRCH_PREFIX_POS +
#define BIRCH_PREFIX_NEG -
#define BIRCH_PREFIX_NOT !

/**
 * @def BIRCH_PREFIX_FORM
 * 
 * Define member variables and functions of a form for a prefix unary
 * operator. This is necessary because a built-in operator `@` cannot be
 * called with the function-like syntax `operator@(x)`, but rather must be
 * called with the prefix notation `@x`.
 */
#define BIRCH_PREFIX_FORM(This, op) \
  Middle m; \
  \
  MEMBIRCH_STRUCT(This) \
  MEMBIRCH_STRUCT_MEMBERS(m) \
  \
  auto operator->() { \
    return this; \
  } \
  \
  const auto operator->() const { \
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
    birch::reset(m); \
  } \
  \
  void relink(const RelinkVisitor& visitor) { \
    birch::relink(m, visitor); \
  } \
  \
  void constant() const { \
    birch::constant(m); \
  } \
  \
  bool isConstant() const { \
    return birch::is_constant(m); \
  } \
  \
  void args(const ArgsVisitor& visitor) { \
    birch::args(m, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) { \
    birch::deep_grad(m, visitor); \
  } \
  \
  auto value() const { \
    return op birch::value(m); \
  } \
  \
  auto eval() const { \
    return op birch::eval(m); \
  } \
  \
  auto peek() const { \
    return op birch::peek(m); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return op birch::move(m, visitor); \
  } \
  \
  auto peg() const { \
    using T = std::decay_t<decltype(birch::peg(m))>; \
    return This<T>{birch::peg(m)}; \
  } \
  auto tag() const { \
    using T = decltype(birch::tag(m)); \
    return This<T>{birch::tag(m)}; \
  }
