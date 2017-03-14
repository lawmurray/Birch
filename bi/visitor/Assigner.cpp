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

bi::Type* bi::Assigner::modify(ModelReference* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(ModelParameter* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(AssignableType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(BracketsType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(ParenthesesType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(RandomType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(LambdaType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(TypeList* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}

bi::Type* bi::Assigner::modify(VariantType* o) {
  Modifier::modify(o);
  o->assignable = true;
  return o;
}
