/**
 * @file
 */
#include "src/visitor/ScopedModifier.hpp"

birch::ScopedModifier::ScopedModifier(Package* currentPackage,
    Class* currentClass) :
    ContextualModifier(currentPackage, currentClass),
    inMember(0),
    inGlobal(0) {
  if (currentPackage) {
    scopes.push_back(currentPackage->scope);
    if (currentClass) {
      scopes.push_back(currentClass->scope);
    }
  }
}

birch::ScopedModifier::~ScopedModifier() {
  //
}

birch::Package* birch::ScopedModifier::modify(Package* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Expression* birch::ScopedModifier::modify(LambdaFunction* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Expression* birch::ScopedModifier::modify(Member* o) {
  o->left = o->left->accept(this);
  ++inMember;
  o->right = o->right->accept(this);
  --inMember;
  return o;
}

birch::Expression* birch::ScopedModifier::modify(Global* o) {
  ++inGlobal;
  o->single = o->single->accept(this);
  --inGlobal;
  return o;
}

birch::Statement* birch::ScopedModifier::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(Function* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(BinaryOperator* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(UnaryOperator* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(Program* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(ConversionOperator* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(SliceOperator* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(Class* o) {
  this->currentClass = o;
  scopes.push_back(o->scope);
  o->typeParams = o->typeParams->accept(this);
  o->base = o->base->accept(this);
  scopes.push_back(o->initScope);
  o->params = o->params->accept(this);
  o->args = o->args->accept(this);
  scopes.pop_back();
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  this->currentClass = nullptr;
  return o;
}

birch::Statement* birch::ScopedModifier::modify(If* o) {
  scopes.push_back(o->scope);
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  scopes.push_back(o->falseScope);
  o->falseBraces = o->falseBraces->accept(this);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(For* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(Parallel* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(While* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(DoWhile* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  o->cond = o->cond->accept(this);
  return o;
}

birch::Statement* birch::ScopedModifier::modify(With* o) {
  scopes.push_back(o->scope);
  ContextualModifier::modify(o);
  scopes.pop_back();
  return o;
}

birch::Statement* birch::ScopedModifier::modify(Block* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  return o;
}
