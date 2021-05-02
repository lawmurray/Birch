/**
 * @file
 */
#include "src/visitor/ContextualModifier.hpp"

birch::ContextualModifier::ContextualModifier(Package* currentPackage,
    Class* currentClass) :
    currentPackage(currentPackage),
    currentClass(currentClass) {
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
