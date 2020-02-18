/**
 * @file
 */
#include "bi/visitor/ScopedModifier.hpp"

bi::ScopedModifier::ScopedModifier() :
    inMember(0),
    inGlobal(0),
    currentClass(nullptr),
    currentFiber(nullptr) {
  //
}

bi::ScopedModifier::~ScopedModifier() {
  //
}

bi::Package* bi::ScopedModifier::modify(Package* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Expression* bi::ScopedModifier::modify(LambdaFunction* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Expression* bi::ScopedModifier::modify(Member* o) {
  o->left = o->left->accept(this);
  ++inMember;
  o->right = o->right->accept(this);
  --inMember;
  return o;
}

bi::Expression* bi::ScopedModifier::modify(Global* o) {
  ++inGlobal;
  o->single = o->single->accept(this);
  --inGlobal;
  return o;
}

bi::Statement* bi::ScopedModifier::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(Function* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(Fiber* o) {
  scopes.push_back(o->scope);
  currentFiber = o;
  Modifier::modify(o);
  currentFiber = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(BinaryOperator* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(UnaryOperator* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(Program* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(ConversionOperator* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(Class* o) {
  scopes.push_back(o->scope);
  o->typeParams = o->typeParams->accept(this);
  o->base = o->base->accept(this);
  scopes.push_back(o->initScope);
  o->params = o->params->accept(this);
  o->args = o->args->accept(this);
  scopes.pop_back();
  currentClass = o;
  o->braces = o->braces->accept(this);
  currentClass = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(If* o) {
  scopes.push_back(o->scope);
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  scopes.push_back(o->falseScope);
  o->falseBraces = o->falseBraces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(For* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(Parallel* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(While* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ScopedModifier::modify(DoWhile* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  o->cond = o->cond->accept(this);
  return o;
}
