/**
 * @file
 */
#include "bi/expression/Global.hpp"

#include "bi/visitor/all.hpp"

bi::Global::Global(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Global::~Global() {
  //
}

bi::Lookup bi::Global::lookup(Expression* args) {
  return single->lookup(args);
}

bi::GlobalVariable* bi::Global::resolve(Call<GlobalVariable>* o) {
  return single->resolve(o);
}

bi::Function* bi::Global::resolve(Call<Function>* o) {
  return single->resolve(o);
}

bi::Fiber* bi::Global::resolve(Call<Fiber>* o) {
  return single->resolve(o);
}

bi::Expression* bi::Global::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Global::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Global::accept(Visitor* visitor) const {
  visitor->visit(this);
}
