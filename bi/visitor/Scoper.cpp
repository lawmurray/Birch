/**
 * @file
 */
#include "bi/visitor/Scoper.hpp"

bi::Scoper::Scoper(Package* currentPackage, Class* currentClass,
    Fiber* currentFiber) :
    ScopedModifier(currentPackage, currentClass, currentFiber) {
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
  /* handle start function, using new Scoper for correct scoping */
  Scoper scoper(currentPackage, currentClass);
  o->start = o->start->accept(&scoper);

  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Fiber* o) {
  /* handle start function, using new Scoper for correct scoping */
  Scoper scoper(currentPackage, currentClass);
  o->start = o->start->accept(&scoper);

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

bi::Statement* bi::Scoper::modify(Yield* o) {
  /* handle resume function, using new Scoper for correct scoping */
  if (o->resume) {
    Scoper scoper(currentPackage, currentClass);
    o->resume = o->resume->accept(&scoper);
  }
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Scoper::modify(Generic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}
