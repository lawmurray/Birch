/**
 * @file
 */
#include "src/visitor/ContextualModifier.hpp"

birch::ContextualModifier::ContextualModifier(Package* currentPackage,
    Class* currentClass, Fiber* currentFiber) :
    currentPackage(currentPackage),
    currentClass(currentClass),
    currentFiber(currentFiber) {
  //
}

birch::ContextualModifier::~ContextualModifier() {
  //
}

birch::Package* birch::ContextualModifier::modify(Package* o) {
  currentPackage = o;
  Modifier::modify(o);
  currentPackage = nullptr;
  return o;
}

birch::Statement* birch::ContextualModifier::modify(Class* o) {
  currentClass = o;
  Modifier::modify(o);
  currentClass = nullptr;
  return o;
}

birch::Statement* birch::ContextualModifier::modify(Fiber* o) {
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  return o;
}

birch::Statement* birch::ContextualModifier::modify(MemberFiber* o) {
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  return o;
}
