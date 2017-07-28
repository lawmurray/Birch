/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Call<bi::UnaryOperator>::Call(shared_ptr<Name> name, Expression* single,
    shared_ptr<Location> loc, const UnaryOperator* target) :
    Expression(loc),
    Named(name),
    Unary<Expression>(single),
    Reference<UnaryOperator>(target) {
  //
}

bi::Call<bi::UnaryOperator>::~Call() {
  //
}

bi::Expression* bi::Call<bi::UnaryOperator>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Call<bi::UnaryOperator>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Call<bi::UnaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Call<bi::UnaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Call<bi::UnaryOperator>::definitely(
    const Call<UnaryOperator>& o) const {
  return single->definitely(*o.single);
}

bool bi::Call<bi::UnaryOperator>::definitely(const UnaryOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::Call<bi::UnaryOperator>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Call<bi::UnaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Call<bi::UnaryOperator>::possibly(
    const Call<UnaryOperator>& o) const {
  return single->possibly(*o.single);
}

bool bi::Call<bi::UnaryOperator>::possibly(const UnaryOperator& o) const {
  return single->possibly(*o.single);
}

bool bi::Call<bi::UnaryOperator>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
