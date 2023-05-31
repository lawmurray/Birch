/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"
#include "birch/form/Binary.hpp"

#define BIRCH_INFIX_ADD +
#define BIRCH_INFIX_SUB -
#define BIRCH_INFIX_MUL *
#define BIRCH_INFIX_DIV /
#define BIRCH_INFIX_AND &&
#define BIRCH_INFIX_OR ||
#define BIRCH_INFIX_EQUAL ==
#define BIRCH_INFIX_NOT_EQUAL !=
#define BIRCH_INFIX_LESS <
#define BIRCH_INFIX_LESS_OR_EQUAL <=
#define BIRCH_INFIX_GREATER >
#define BIRCH_INFIX_GREATER_OR_EQUAL >=

/**
 * @def BIRCH_INFIX_FORM
 * 
 * Define member variables and functions of a form for an infix binary
 * operator. This is necessary because a built-in operator `@` cannot be
 * called with the function-like syntax `operator@(x, y)`, but rather must be
 * called with the infix notation `x @ y`.
 */
#define BIRCH_INFIX_FORM(This, op) \
  Left l; \
  Right r; \
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
  void args(const ArgsVisitor& visitor) { \
    birch::args(l, visitor); \
    birch::args(r, visitor); \
  } \
  \
  void deepGrad(const GradVisitor& visitor) { \
    birch::deep_grad(l, visitor); \
    birch::deep_grad(r, visitor); \
  } \
  \
  auto value() const { \
    return birch::value(l) op birch::value(r); \
  } \
  \
  auto eval() const { \
    return birch::eval(l) op birch::eval(r); \
  } \
  \
  auto peek() const { \
    return birch::peek(l) op birch::peek(r); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return birch::move(l, visitor) op birch::move(r, visitor); \
  } \
  \
  auto peg() const { \
    using T = std::decay_t<decltype(birch::peg(l))>; \
    using U = std::decay_t<decltype(birch::peg(r))>; \
    return This<T,U>{birch::peg(l), birch::peg(r)}; \
  } \
  auto tag() const { \
    using T = decltype(birch::tag(l)); \
    using U = decltype(birch::tag(r)); \
    return This<T,U>{birch::tag(l), birch::tag(r)}; \
  }
