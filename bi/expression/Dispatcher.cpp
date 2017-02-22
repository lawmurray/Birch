/**
 * @file
 */
#include "bi/expression/Dispatcher.hpp"

#include "bi/visitor/all.hpp"

bi::Dispatcher::Dispatcher(shared_ptr<Name> name, shared_ptr<Name> mangled,
    shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    mangled(mangled) {
  this->arg = this;
}

bi::Dispatcher::~Dispatcher() {
  //
}

void bi::Dispatcher::insert(FuncParameter* func) {
  /* pre-condition */
  assert(*func->name == *name);
  assert(*func->mangled == *mangled);

  funcs.insert(func);
}

bi::Expression* bi::Dispatcher::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Dispatcher::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Dispatcher::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Dispatcher::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::Dispatcher::definitely(Dispatcher& o) {
  /* pre-conditions */
  assert(funcs.size() > 0);
  assert(o.funcs.size() > 0);

  /* comparing any function in one dispatcher to any function in the other
   * dispatcher is sufficient, as all should yield the same result */
  return funcs.any()->definitely(*o.funcs.any()) && o.capture(this);
}

bool bi::Dispatcher::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::Dispatcher::possibly(Dispatcher& o) {
  /* pre-conditions */
  assert(funcs.size() > 0);
  assert(o.funcs.size() > 0);

  /* comparing any function in one dispatcher to any function in the other
   * dispatcher is sufficient, as all should yield the same result */
  return funcs.any()->possibly(*o.funcs.any()) && o.capture(this);
}
