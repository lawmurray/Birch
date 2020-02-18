/**
 * @file
 */
#include "bi/visitor/Scoper.hpp"

bi::Scoper::Scoper() {
  //
}

bi::Scoper::~Scoper() {
  //
}

bi::Statement* bi::Scoper::modify(MemberVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(GlobalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(MemberFunction* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Function* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(MemberFiber* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Fiber* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(BinaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(UnaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Program* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Basic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Class* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Scoper::modify(Generic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}
