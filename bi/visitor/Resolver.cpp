/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/exception/all.hpp"

#include <sstream>

bi::Resolver::Resolver() :
    memberScope(nullptr),
    inInputs(0) {
  //
}

bi::Resolver::~Resolver() {
  //
}

void bi::Resolver::modify(File* o) {
  if (o->state == File::RESOLVING) {
    throw CyclicImportException(o);
  } else if (o->state == File::UNRESOLVED) {
    o->state = File::RESOLVING;
    o->scope = new Scope();
    files.push(o);
    push(o->scope.get());
    o->root = o->root->accept(this);
    undefer();
    pop();
    files.pop();
    o->state = File::RESOLVED;
  }
}

bi::Expression* bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type->accept(&cloner),
      o->tail->type->accept(&cloner));
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(ParenthesesExpression* o) {
  Modifier::modify(o);
  o->type = new ParenthesesType(o->single->type->accept(&cloner));
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Index* o) {
  Modifier::modify(o);
  o->type = o->single->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Range* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Member* o) {
  o->left = o->left->accept(this);
  TypeReference* ref = dynamic_cast<TypeReference*>(o->left->type->strip());
  if (ref) {
    assert(ref->target);
    memberScope = ref->target->scope.get();
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type->accept(&cloner)->accept(this);
  o->member = o->right->isMember();

  return o;
}

bi::Expression* bi::Resolver::modify(This* o) {
  if (!type()) {
    throw ThisException(o);
  } else {
    Modifier::modify(o);
    o->type = new TypeReference(type()->name);
    o->type->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(BracketsExpression* o) {
  Modifier::modify(o);

  const int typeSize = o->single->type->count();
  const int indexSize = o->brackets->tupleSize();
  const int rangeDims = o->brackets->tupleDims();
  assert(typeSize == indexSize);  ///@todo Exception

  BracketsType* type = dynamic_cast<BracketsType*>(o->single->type->strip());
  assert(type);
  if (rangeDims > 0) {
    o->type = new BracketsType(type->single->accept(&cloner), rangeDims);
    o->type = o->type->accept(this);
    if (o->single->type->assignable) {
      o->single->type->accept(&assigner);
    }
  } else {
    o->type = type->single->accept(&cloner)->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(VarReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(FuncReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  if (o->isAssign() && *o->name == "<-") {
    if (!o->getLeft()->type->assignable) {
      throw NotAssignableException(o);
    } else if (!o->getRight()->type->definitely(*o->getLeft()->type)) {
      ///@todo Problems with assignable in comparison here
      //resolve(o, memberScope);
      //o->type = o->target->type->accept(&cloner)->accept(this);
    }
  } else {
    resolve(o, memberScope);
    o->type = o->target->type->accept(&cloner)->accept(this);
    //o->type->assignable = false;  // rvalue
  }
  
  return o;
}

bi::Type* bi::Resolver::modify(TypeReference* o) {
  Scope* memberScope = takeMemberScope();
  assert(!memberScope);

  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Expression* bi::Resolver::modify(VarParameter* o) {
  Modifier::modify(o);
  if (!inInputs) {
    o->type->accept(&assigner);
  }
  if (!o->name->isEmpty()) {
    top()->add(o);
  }
  if (o->type->isLambda()) {
    o->func = makeLambda(o)->accept(this);
  }
  if (!o->value->isEmpty()) {
    if (!o->type->assignable) {
      throw NotAssignableException(o);
    } else if (!o->value->type->definitely(*o->type)) {
      //throw InvalidAssignmentException(o);
    }
  }
  return o;
}

bi::Expression* bi::Resolver::modify(FuncParameter* o) {
  push();
  if (o->isAssign()) {
    o->getLeft()->type->accept(&assigner);
  }
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->type = o->type->accept(this)->accept(&assigner);
  if (o->isLambda()) {
    o->braces = o->braces->accept(this);
  } else if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();

  if (!o->name->isEmpty()) {
    top()->add(o);
  }

  return o;
}

bi::Expression* bi::Resolver::modify(ConversionParameter* o) {
  push();
  o->type = o->type->accept(this)->accept(&assigner);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Prog* bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Type* bi::Resolver::modify(TypeParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  o->base = o->base->accept(this);
  ///@todo Check that the type and its base are both struct or both class
  o->scope = pop();

  if (!o->base->isEmpty()) {
    o->scope->inherit(o->super()->scope.get());
  }

  top()->add(o);
  push(o->scope.get());
  types.push(o);
  o->braces = o->braces->accept(this);
  types.pop();
  pop();

  return o;
}

bi::Statement* bi::Resolver::modify(Import* o) {
  o->file->accept(this);
  top()->import(o->file->scope.get());
  return o;
}

bi::Statement* bi::Resolver::modify(If* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::Resolver::modify(For* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that index, from and to are of type Integer
  return o;
}

bi::Statement* bi::Resolver::modify(While* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::Resolver::modify(Return* o) {
  Modifier::modify(o);
  ///@todo Check that the type of the expression is correct
  return o;
}

bi::FuncParameter* bi::Resolver::makeLambda(VarParameter* o) {
  LambdaType* lambda = dynamic_cast<LambdaType*>(o->type->strip());
  assert(lambda);

  /* parameters */
  Expression* parens;
  std::list<const Type*> types;
  for (auto iter = lambda->parens->begin(); iter != lambda->parens->end();
      ++iter) {
    types.push_back(*iter);
  }
  if (types.size() > 0) {
    auto iter = types.rbegin();
    parens = new VarParameter(new Name(), (*iter)->accept(&cloner));
    ++iter;
    while (iter != types.rend()) {
      parens = new ExpressionList(
          new VarParameter(new Name(), (*iter)->accept(&cloner)), parens);
      ++iter;
    }
  } else {
    parens = new EmptyExpression();
  }

  /* return type */
  Type* type = lambda->type->accept(&cloner);

  /* function */
  FuncParameter* func = new FuncParameter(o->name, parens, type,
      new EmptyExpression(), LAMBDA_FUNCTION);

  return func;
}

bi::Scope* bi::Resolver::takeMemberScope() {
  Scope* scope = memberScope;
  memberScope = nullptr;
  return scope;
}

bi::Scope* bi::Resolver::top() {
  return scopes.back();
}

bi::Scope* bi::Resolver::bottom() {
  return scopes.front();
}

void bi::Resolver::push(Scope* scope) {
  if (scope) {
    scopes.push_back(scope);
  } else {
    scopes.push_back(new Scope());
  }
}

bi::Scope* bi::Resolver::pop() {
  /* pre-conditions */
  assert(scopes.size() > 0);

  Scope* res = scopes.back();
  scopes.pop_back();
  return res;
}

void bi::Resolver::resolve(VarReference* ref, Scope* scope) {
  if (scope) {
    /* use provided scope, usually a membership scope */
    scope->resolve(ref);
  } else {
    /* use current stack of scopes */
    ref->target = nullptr;
    for (auto iter = scopes.rbegin(); !ref->target && iter != scopes.rend();
        ++iter) {
      (*iter)->resolve(ref);
    }
  }
  if (!ref->target) {
    throw UnresolvedReferenceException(ref);
  } else {
    ref->member = ref->target->isMember();
  }
}

void bi::Resolver::resolve(FuncReference* ref, Scope* scope) {
  if (scope) {
    /* use provided scope, usually a membership scope */
    scope->resolve(ref);
  } else {
    /* use current stack of scopes */
    ref->target = nullptr;
    for (auto iter = scopes.rbegin(); !ref->target && iter != scopes.rend();
        ++iter) {
      (*iter)->resolve(ref);
    }
  }
  if (!ref->target) {
    throw UnresolvedReferenceException(ref);
  }
}

void bi::Resolver::resolve(TypeReference* ref) {
  ref->target = nullptr;
  for (auto iter = scopes.rbegin(); !ref->target && iter != scopes.rend();
      ++iter) {
    (*iter)->resolve(ref);
  }
  if (!ref->target) {
    throw UnresolvedReferenceException(ref);
  }
}

void bi::Resolver::defer(Expression* o) {
  if (files.size() == 1) {
    /* ignore bodies in imported files */
    defers.push_back(std::make_tuple(o, top(), type()));
  }
}

void bi::Resolver::undefer() {
  auto iter = defers.begin();
  while (iter != defers.end()) {
    auto o = std::get<0>(*iter);
    auto scope = std::get<1>(*iter);
    auto type = std::get<2>(*iter);

    if (type) {
      push(type->scope.get());
    }
    push(scope);
    types.push(type);
    o->accept(this);
    types.pop();
    pop();
    if (type) {
      pop();
    }
    ++iter;
  }
  defers.clear();
}

bi::TypeParameter* bi::Resolver::type() {
  if (types.empty()) {
    return nullptr;
  } else {
    return types.top();
  }
}
