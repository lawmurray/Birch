/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class To, class Middle>
struct Cast {
  Middle m;
 
  MEMBIRCH_STRUCT(Cast)
  MEMBIRCH_STRUCT_MEMBERS(m)
  BIRCH_FORM

  auto operator->() {
    return this;
  }

  auto operator->() const {
    return this;
  }

  operator auto() const {
    return value();
  }

  auto operator*() const {
    return wait(value());
  }

  void reset() {
    birch::reset(m);
  }
 
  void relink(const RelinkVisitor& visitor) {
    birch::relink(m, visitor);
  }
 
  void constant() const {
    birch::constant(m);
  }
 
  bool isConstant() const {
    return birch::is_constant(m);
  }
 
  void args(const ArgsVisitor& visitor) {
    birch::args(m, visitor);
  }
 
  void deepGrad(const GradVisitor& visitor) {
    birch::deep_grad(m, visitor);
  }
 
  auto value() const {
    return numbirch::cast<To>(birch::value(m));
  }
 
  auto eval() const {
    return numbirch::cast<To>(birch::eval(m));
  }
 
  auto peek() const {
    return numbirch::cast<To>(birch::peek(m));
  }
 
  auto move(const MoveVisitor& visitor) const {
    return numbirch::cast<To>(birch::move(m, visitor));
  }
 
  auto peg() const {
    using T = std::decay_t<decltype(birch::peg(m))>;
    return Cast<To,T>{birch::peg(m)};
  }

  auto tag() const {
    using T = decltype(birch::tag(m));
    return Cast<To,T>{birch::tag(m)};
  }

  template<class G>
  void shallowGrad(const G& g, const GradVisitor& visitor) {
    birch::shallow_grad(m, cast_grad<To>(g, birch::peek(*this),
        birch::peek(m)), visitor);
  }
};

template<class To, class Middle>
auto cast(const Middle& m) {
  return Cast<To,decltype(tag(m))>{tag(m)};
}

}
