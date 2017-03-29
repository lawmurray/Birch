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
    if (o->dispatcher) {
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
    LambdaType* lambda = dynamic_cast<LambdaType*>(o->type->strip());
    assert(lambda);
    Expression* parens = new ParenthesesExpression();
    Expression* result = new VarParameter(new Name(),
        lambda->result->accept(&cloner));
    o->func = new FuncParameter(o->name, parens, result,
        new EmptyExpression(), LAMBDA);
    o->func = o->func->accept(this);
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
  } else if (!o->braces->isEmpty() ) {
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

bi::Dispatcher* bi::Resolver::modify(Dispatcher* o) {
  /* pre-condition */
  assert(o->funcs.size() > 0);

  /* initialise with return type of first function */
  auto iter = o->funcs.rbegin();
  o->type = (*iter)->result->type->accept(&cloner)->accept(this);
  ++iter;

  /* fill with types of remaining functions */
  while (iter != o->funcs.rend()) {
    FuncParameter* func = *iter;
    o->type = combine(func->result->type.get(), o->type.release());
    ++iter;
  }

  return o;
}

bi::Expression* bi::Resolver::parameters(Expression* parens1,
    Expression* parens2) {
  /*ArgumentCapturer capturer(parens1, parens2);
   Expression* parens3 = parens2->accept(&cloner);
   Gatherer<VarParameter> gatherer;
   parens3->accept(&gatherer);*/

  /* replace types of parameters with types of arguments */
  /*auto iter2 = capturer.begin();
   auto iter3 = gatherer.begin();
   while (iter2 != capturer.end() && iter3 != gatherer.end()) {
   (*iter3)->type = iter2->first->type->accept(&cloner);
   ++iter2;
   ++iter3;
   }
   assert(iter2 == capturer.end());
   assert(iter3 == gatherer.end());

   return parens3;*/
}

bi::Type* bi::Resolver::combine(Type* o1, Type* o2) {
  VariantType* variant = dynamic_cast<VariantType*>(o2);
  if (variant) {
    /* this parameter already has a variant type, add the type of the
     * argument to this */
    variant->add(o1);
    return variant;
  } else if (!o1->equals(*o2)) {
    /* make a new variant type, assuming current type is definite */
    variant = new VariantType(o2);
    variant->add(o1);
    return variant;
  }
  return o2;
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

void bi::Resolver::resolve(FuncReference* ref, Scope* scope) {
  resolve<FuncReference>(ref, scope);
  if (ref->possibles.size() > 0) {
    FuncParameter* param = ref->target;
    Dispatcher* dispatcher = new Dispatcher(param->name);
    dispatcher->push_front(param);
    for (auto iter = ref->possibles.rbegin(); iter != ref->possibles.rend();
        ++iter) {
      dispatcher->push_front(*iter);
    }
    dispatcher = dispatcher->accept(this);
    if (bottom()->contains(dispatcher)) {
      /* reuse identical dispatcher in the scope */
      Dispatcher* existing = bottom()->get(dispatcher);
      delete dispatcher;
      dispatcher = existing;
    } else {
      /* add current dispatcher to the scope */
      bottom()->add(dispatcher);
    }
    ref->dispatcher = dispatcher;
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
template void bi::Resolver::resolve(ModelReference* ref, Scope* scope);
