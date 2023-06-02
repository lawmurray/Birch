/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class To, class Middle>
struct Cast {
  BIRCH_UNARY_FORM(Cast)
};

template<class To, class Middle>
struct is_form<Cast<To,Middle>> {
  static constexpr bool value = true;
};

template<class To, class Middle>
int rows(const Cast<To,Middle>& o) {
  return rows(o.m);
}

template<class To, class Middle>
int columns(const Cast<To,Middle>& o) {
  return columns(o.m);
}

template<class To, class Middle>
int length(const Cast<To,Middle>& o) {
  return length(o.m);
}

template<class To, class Middle>
int size(const Cast<To,Middle>& o) {
  return size(o.m);
}

template<class To, class Middle>
auto value(const Cast<To,Middle>& o) {
  return numbirch::cast<To>(value(o.m));
}

template<class To, class Middle>
auto eval(const Cast<To,Middle>& o) {
  return numbirch::cast<To>(eval(o.m));
}

template<class To, class Middle>
auto peek(const Cast<To,Middle>& o) {
  return numbirch::cast<To>(peek(o.m));
}

template<class To, class Middle>
auto move(const Cast<To,Middle>& o, const MoveVisitor& visitor) {
  return numbirch::cast<To>(move(o.m, visitor));
}

template<class To, class Middle>
auto peg(const Cast<To,Middle>& o) {
  using T = std::decay_t<decltype(peg(o.m))>;
  return Cast<To,T>{peg(o.m)};
}
template<class To, class Middle>
auto tag(const Cast<To,Middle>& o) {
  using T = decltype(tag(o.m));
  return Cast<To,T>{tag(o.m)};
}

template<class To, class Middle>
void reset(Cast<To,Middle>& o) {
  reset(o.m);
}

template<class To, class Middle>
void relink(Cast<To,Middle>& o, const RelinkVisitor& visitor) {
  relink(o.m, visitor);
}

template<class To, class Middle>
void constant(const Cast<To,Middle>& o) {
  constant(o.m);
}

template<class To, class Middle>
bool is_constant(const Cast<To,Middle>& o) {
  return is_constant(o.m);
}

template<class To, class Middle>
void args(Cast<To,Middle>& o, const ArgsVisitor& visitor) {
  args(o.m, visitor);
}

template<class To, class Middle>
void deep_grad(Cast<To,Middle>& o, const GradVisitor& visitor) {
  deep_grad(o.m, visitor);
}

template<class To, class Middle, class G>
void shallow_grad(Cast<To,Middle>& o, const G& g, const GradVisitor& visitor) {
  shallow_grad(o.m, cast_grad<To>(g, peek(o), peek(o.m)), visitor);
}

template<class To, class Middle>
Cast<To,Middle>::operator auto() const {
  return numbirch::cast<To>(value(m));
}

template<class To, class Middle>
auto Cast<To,Middle>::operator*() const { 
  return wait(numbirch::cast<To>(value(m)));
}

template<class To, class Middle>
auto cast(const Middle& m) {
  return Cast<To,decltype(tag(m))>{tag(m)};
}

}
