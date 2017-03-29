/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/exception/all.hpp"

#include <sstream>

bi::Resolver::Resolver() :
    membershipScope(nullptr),
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

  ModelReference* ref = dynamic_cast<ModelReference*>(o->left->type->strip());
  if (ref) {
    membershipScope = ref->target->scope.get();
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type->accept(&cloner)->accept(this);
  if (o->left->type->assignable) {
    o->type->accept(&assigner);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(This* o) {
  if (!model()) {
    throw ThisException(o);
  } else {
    Modifier::modify(o);
    o->type = new ModelReference(model()->name, new EmptyExpression(),
        nullptr, model());
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
  Scope* membershipScope = takeMembershipScope();
  Modifier::modify(o);
  resolve(o, membershipScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(FuncReference* o) {
  Scope* membershipScope = takeMembershipScope();
  Modifier::modify(o);
  if (o->isAssign()) {
    if (!o->getLeft()->type->assignable) {
      throw NotAssignableException(o);
    } else if (!o->getRight()->type->definitely(*o->getLeft()->type)) {
      //throw InvalidAssignmentException(o);
    }
  } else {
    resolve(o, membershipScope);
    if (o->possibles.size() > 0) {
      o->dispatcher = makeDispatcher(o);
      o->type = o->dispatcher->type->accept(&cloner)->accept(this);
    } else {
      o->type = o->target->type->accept(&cloner)->accept(this);
    }
    o->type->assignable = false;  // rvalue
  }
  return o;
}

bi::Type* bi::Resolver::modify(ModelReference* o) {
  Scope* membershipScope = takeMembershipScope();
  Modifier::modify(o);
  resolve(o, membershipScope);
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
    o->func = makeLambda(o);
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
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->result = o->result->accept(this)->accept(&assigner);
  o->type = o->result->type->accept(&cloner)->accept(this);
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

bi::Prog* bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Type* bi::Resolver::modify(ModelParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  o->base = o->base->accept(this);
  o->scope = pop();
  top()->add(o);
  push(o->scope.get());
  models.push(o);
  o->braces = o->braces->accept(this);
  models.pop();
  pop();

  return o;
}

bi::Statement* bi::Resolver::modify(Import* o) {
  o->file->accept(this);
  top()->import(o->file->scope.get());
  return o;
}

bi::Statement* bi::Resolver::modify(Conditional* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::Resolver::modify(Loop* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Type* bi::Resolver::modify(AssignableType* o) {
  Modifier::modify(o);
  o->accept(&assigner);
  return o;
}

bi::FuncParameter* bi::Resolver::makeLambda(VarParameter* o) {
  LambdaType* type = dynamic_cast<LambdaType*>(o->type.get());
  assert(type);

  /* parameters */
  Expression* parens;
  std::list<const Type*> types;
  for (auto iter = type->parens->begin(); iter != type->parens->end();
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

  /* result */
  Expression* result = new VarParameter(new Name(),
      type->result->accept(&cloner));

  /* function */
  FuncParameter* func = new FuncParameter(o->name, parens, result,
      new EmptyExpression(), LAMBDA);
  func->accept(this);

  return func;
}

bi::Dispatcher* bi::Resolver::makeDispatcher(FuncReference* o) {
  Dispatcher* dispatcher = new Dispatcher(o);

  /* reuse an existing, identical dispatcher in the scope if possible */
  if (bottom()->contains(dispatcher)) {
    Dispatcher* existing = bottom()->get(dispatcher);
    delete dispatcher;
    dispatcher = existing;
  } else {
    bottom()->add(dispatcher);
  }

  return dispatcher;
}

bi::Scope* bi::Resolver::takeMembershipScope() {
  Scope* scope = membershipScope;
  membershipScope = nullptr;
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

template<class ReferenceType>
void bi::Resolver::resolve(ReferenceType* ref, Scope* scope) {
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

void bi::Resolver::defer(Expression* o) {
  if (files.size() == 1) {
    /* ignore bodies in imported files */
    defers.push_back(std::make_tuple(o, top(), model()));
  }
}

void bi::Resolver::undefer() {
  auto iter = defers.begin();
  while (iter != defers.end()) {
    auto o = std::get<0>(*iter);
    auto scope = std::get<1>(*iter);
    auto model = std::get<2>(*iter);

    push(scope);
    models.push(model);
    o->accept(this);
    models.pop();
    pop();
    ++iter;
  }
  defers.clear();
}

bi::ModelParameter* bi::Resolver::model() {
  if (models.empty()) {
    return nullptr;
  } else {
    return models.top();
  }
}

template void bi::Resolver::resolve(VarReference* ref, Scope* scope);
template void bi::Resolver::resolve(FuncReference* ref, Scope* scope);
template void bi::Resolver::resolve(ModelReference* ref, Scope* scope);
