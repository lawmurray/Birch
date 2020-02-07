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

bi::Expression* bi::Scoper::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
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
  o->yield->accept(this);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Fiber* o) {
  scopes.back()->add(o);
  o->yield->accept(this);
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

bi::Statement* bi::Scoper::modify(Yield* o) {
  if (o->resume) {
    o->resume->accept(this);
  }
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Scoper::modify(Generic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}
