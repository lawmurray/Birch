/**
 * @file
 */
#include "bi/visitor/Annotator.hpp"

bi::Annotator::Annotator(const Annotation a) : a(a) {
  //
}

bi::Annotator::~Annotator() {
  //
}

bi::Expression* bi::Annotator::modify(Identifier<Parameter>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<FiberParameter>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<GlobalVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<MemberVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<FiberVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<LocalVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<ParallelVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(Identifier<ForVariable>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<Unknown>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<Function>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<Fiber>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<MemberFiber>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<MemberFunction>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<BinaryOperator>* o) {
  o->target->set(a);
  return o;
}

bi::Expression* bi::Annotator::modify(OverloadedIdentifier<UnaryOperator>* o) {
  o->target->set(a);
  return o;
}

bi::Type* bi::Annotator::modify(ClassType* o) {
  o->target->set(a);
  return o;
}

bi::Type* bi::Annotator::modify(BasicType* o) {
  o->target->set(a);
  return o;
}

bi::Type* bi::Annotator::modify(GenericType* o) {
  o->target->set(a);
  return o;
}
