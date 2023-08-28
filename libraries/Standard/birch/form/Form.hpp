/**
 * @file
 */
#pragma once

#include "birch/form/Empty.hpp"

#define BIRCH_FORM \
  auto operator->() { \
    return this; \
  } \
  auto operator->() const { \
    return this; \
  } \
  operator auto() const { \
    return value(); \
  } \
  auto operator*() const { \
    return wait(value()); \
  } \
  auto value() const { \
    this->constant(); \
    return eval(); \
  } \
  template<class Buffer> \
  void write(const Buffer& buffer) const { \
    buffer->set(value()); \
  } \
  template<class Buffer> \
  void write(const Integer t, const Buffer& buffer) const { \
    buffer->set(value()); \
  }

namespace birch {

template<argument T = Empty, argument U = Empty, class V = Empty,
    class W = Empty, class X = Empty>
struct Form {
  using T1 = T;
  using U1 = U;
  using V1 = V;
  using W1 = W;
  using X1 = X;

  [[no_unique_address]] T x;
  [[no_unique_address]] U y;
  [[no_unique_address]] V z;
  [[no_unique_address]] W a;
  [[no_unique_address]] X b;
 
  MEMBIRCH_STRUCT(Form)
  MEMBIRCH_STRUCT_MEMBERS(x, y, z, a, b)

  void reset() {
    birch::reset(x);
    birch::reset(y);
    birch::reset(z);
    birch::reset(a);
    birch::reset(b);
  }
 
  void relink(const RelinkVisitor& visitor) {
    birch::relink(x, visitor);
    birch::relink(y, visitor);
    birch::relink(z, visitor);
    birch::relink(a, visitor);
    birch::relink(b, visitor);
  }
 
  void constant() const {
    birch::constant(x);
    birch::constant(y);
    birch::constant(z);
    birch::constant(a);
    birch::constant(b);
  }
 
  bool isConstant() const {
    return is_constant(x) &&
        is_constant(y) &&
        is_constant(z) &&
        is_constant(a) &&
        is_constant(b);
 }
 
  void move(const MoveVisitor& visitor) const {
    birch::move(x, visitor);
    birch::move(y, visitor);
    birch::move(z, visitor);
    birch::move(a, visitor);
    birch::move(b, visitor);
  }

  void args(const ArgsVisitor& visitor) const {
    birch::args(x, visitor);
    birch::args(y, visitor);
    birch::args(z, visitor);
    birch::args(a, visitor);
    birch::args(b, visitor);
  }

  void deepGrad(const GradVisitor& visitor) const {
    birch::deep_grad(x, visitor);
    birch::deep_grad(y, visitor);
    birch::deep_grad(z, visitor);
    birch::deep_grad(a, visitor);
    birch::deep_grad(b, visitor);
  }
};

}
