/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {
/**
 * Delayed form with one argument.
 */
template<class Middle>
struct Unary : public Form {
  template<class T>
  Unary(T&& m) :
      Form(),
      m(m) {
    //
  }

  void reset() {
    birch::reset(m);
  }

  void relink(const RelinkVisitor& visitor) {
    birch::relink(m, visitor);
  }

  void constant() {
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

  /**
   * Argument.
   */
  Middle m;

  MEMBIRCH_STRUCT(Unary, Form)
  MEMBIRCH_STRUCT_MEMBERS(m)
};

}

#define BIRCH_UNARY_FORM(f, ...) \
  using Value = decltype(f(birch::eval(std::declval<Middle>()), \
      ##__VA_ARGS__)); \
  std::optional<Value> x; \
  \
  void clear() { \
    x.reset(); \
  } \
  \
  auto value() { \
    auto x = this->eval(); \
    this->constant(); \
    return x; \
  } \
  \
  auto eval() const { \
    auto m = birch::eval(this->m); \
    return f(m, ##__VA_ARGS__); \
  } \
  \
  auto peek() { \
    if (!x) { \
      auto m = birch::peek(this->m); \
      this->x = f(m, ##__VA_ARGS__);\
    } \
    return *x; \
  } \
  \
  auto move(const MoveVisitor& visitor) { \
    auto m = birch::move(this->m, visitor); \
    return f(m, ##__VA_ARGS__); \
  }

#define BIRCH_UNARY_GRAD(f_grad, ...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    auto x = birch::peek(*this); \
    auto m = birch::peek(this->m); \
    birch::shallow_grad(this->m, f_grad(g, x, m, ##__VA_ARGS__), visitor); \
    clear(); \
  }
