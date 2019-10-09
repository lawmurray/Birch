/**
 * @file
 */
#include "bi/expression/Member.hpp"

#include "bi/visitor/all.hpp"

bi::Member::Member(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bi::Member::~Member() {
  //
}

bool bi::Member::isAssignable() const {
  return right->isAssignable();
}

bi::Lookup bi::Member::lookup(Expression* args) {
  return right->lookup(args);
}

bi::MemberVariable* bi::Member::resolve(Call<MemberVariable>* o) {
  return right->resolve(o);
}

bi::MemberFunction* bi::Member::resolve(Call<MemberFunction>* o) {
  return right->resolve(o);
}

bi::MemberFiber* bi::Member::resolve(Call<MemberFiber>* o) {
  return right->resolve(o);
}

bi::Expression* bi::Member::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Member::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Member::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
