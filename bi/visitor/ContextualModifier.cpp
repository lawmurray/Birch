/**
 * @file
 */
#include "bi/visitor/ContextualModifier.hpp"

bi::ContextualModifier::ContextualModifier(Package* currentPackage,
    Class* currentClass, Fiber* currentFiber) :
    currentPackage(currentPackage),
    currentClass(currentClass),
    currentFiber(currentFiber) {
  //
}

bi::ContextualModifier::~ContextualModifier() {
  //
}

bi::Package* bi::ContextualModifier::modify(Package* o) {
  currentPackage = o;
  Modifier::modify(o);
  currentPackage = nullptr;
  return o;
}

bi::Statement* bi::ContextualModifier::modify(Class* o) {
  currentClass = o;
  Modifier::modify(o);
  currentClass = nullptr;
  return o;
}

bi::Statement* bi::ContextualModifier::modify(Fiber* o) {
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  return o;
}

bi::Statement* bi::ContextualModifier::modify(MemberFiber* o) {
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  return o;
}
