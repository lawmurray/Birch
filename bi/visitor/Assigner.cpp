/**
 * @file
 */
#include "bi/visitor/Assigner.hpp"

bi::Assigner::~Assigner() {
  //
}

bi::Type* bi::Assigner::modify(EmptyType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(ArrayType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(TupleType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(BinaryType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(FunctionType* o) {
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(OverloadedType* o) {
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(FiberType* o) {
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(OptionalType* o) {
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(ListType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(BasicType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(ClassType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(AliasType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}
