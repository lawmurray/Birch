/**
 * @file
 */
#include "src/visitor/Scoper.hpp"

birch::Scoper::Scoper(Package* currentPackage, Class* currentClass) :
    ScopedModifier(currentPackage, currentClass) {
  //
}

birch::Scoper::~Scoper() {
  //
}

birch::Statement* birch::Scoper::modify(MemberVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(GlobalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(MemberFunction* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(Function* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(BinaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(UnaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(Program* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(Basic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Scoper::modify(Class* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Expression* birch::Scoper::modify(Generic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}
