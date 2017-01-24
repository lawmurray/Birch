/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/exception/all.hpp"

bi::Resolver::Resolver() :
    inInputs(false),
    membershipScope(nullptr) {
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

bi::Expression* bi::Resolver::modify(Range* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Member* o) {
  o->left = o->left->accept(this);

  ModelReference* ref = dynamic_cast<ModelReference*>(o->left->type.get());
  if (ref) {
    membershipScope = ref->target->scope.get();
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->right->type->assignable = o->left->type->assignable;
  o->type = o->right->type->accept(&cloner)->accept(this);
  o->type->assignable = o->right->type->assignable;
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

bi::Expression* bi::Resolver::modify(RandomInit* o) {
  Modifier::modify(o);
  if (!inInputs && !o->left->type->assignable) {
    throw NotAssignableException(o->left.get());
  }
  o->type = o->left->type->accept(&cloner)->accept(this);

  if (!inInputs) {
    o->pull = new FuncReference(o->left->accept(&cloner), new Name("<~"),
        o->right->accept(&cloner));
    o->pull->accept(this);

    o->push = new FuncReference(o->left->accept(&cloner), new Name("~>"),
        o->right->accept(&cloner));
    o->push->accept(this);
  }

  return o;
}

bi::Expression* bi::Resolver::modify(BracketsExpression* o) {
  Modifier::modify(o);

  const int typeSize = o->single->type->count();
  const int indexSize = o->brackets->tupleSize();
  const int indexDims = o->brackets->tupleDims();
  assert(typeSize == indexSize);  ///@todo Exception

  BracketsType* type = dynamic_cast<BracketsType*>(o->single->type.get());
  assert(type);
  o->type = new BracketsType(type->single->accept(&cloner), indexDims);
  o->type = o->type->accept(this);

  return o;
}

bi::Expression* bi::Resolver::modify(VarReference* o) {
  Scope* membershipScope = takeMembershipScope();
  Modifier::modify(o);
  resolve(o, membershipScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  o->type->assignable = o->target->type->assignable;

  return o;
}

bi::Expression* bi::Resolver::modify(FuncReference* o) {
  Scope* membershipScope = takeMembershipScope();
  Modifier::modify(o);
  resolve(o, membershipScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  o->form = o->target->form;

  if (o->isAssignment()) {
    if (inInputs) {
      o->getLeft()->type->assignable = true;
    } else if (!o->getLeft()->type->assignable) {
      throw NotAssignableException(o);
    }
  }

  Gatherer<VarParameter> gatherer;
  o->target->parens->accept(&gatherer);
  for (auto iter = gatherer.gathered.begin(); iter != gatherer.gathered.end();
      ++iter) {
    o->args.push_back((*iter)->arg);
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
  if (!inInputs || dynamic_cast<RandomType*>(o->type->strip())) {
    o->type->assignable = true;
  }
  if (!o->name->isEmpty()) {
    top()->add(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(FuncParameter* o) {
  push();
  inInputs = true;
  o->parens = o->parens->accept(this);
  inInputs = false;
  o->result = o->result->accept(this);
  o->type = o->result->type->accept(&cloner)->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  if (o->isAssignment()) {
    o->getLeft()->type->assignable = true;
  }

  Gatherer<VarParameter> gatherer1;
  o->parens->accept(&gatherer1);
  o->inputs = gatherer1.gathered;

  Gatherer<VarParameter> gatherer2;
  o->result->accept(&gatherer2);
  o->outputs = gatherer2.gathered;

  return o;
}

bi::Prog* bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  Gatherer<VarParameter> gatherer1;
  o->parens->accept(&gatherer1);
  o->inputs = gatherer1.gathered;

  return o;
}

bi::Type* bi::Resolver::modify(ModelParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  o->base = o->base->accept(this);
  models.push(o);
  o->braces = o->braces->accept(this);
  models.pop();
  o->scope = pop();
  top()->add(o);

  if (*o->op != "=") {
    /* create constructor */
    //Expression* parens1 = o->parens->accept(&cloner);
    ///VarParameter* result1 = new VarParameter(new Name(),
    //    new ModelReference(o->name, 0, o));
    //o->constructor = new FuncParameter(o->name, parens1, result1,
    //    new EmptyExpression(), CONSTRUCTOR);
    //o->constructor =
    //    dynamic_cast<FuncParameter*>(o->constructor->accept(this));
    //assert(o->constructor);

    /* create assignment operator */
    Expression* right = new VarParameter(new Name(), new ModelReference(o));
    Expression* left = new VarParameter(new Name(), new ModelReference(o));
    Expression* parens2 = new ParenthesesExpression(
        new ExpressionList(left, right));
    Expression* result2 = new VarParameter(new Name(), new ModelReference(o));
    o->assignment = new FuncParameter(new Name("<-"), parens2, result2,
        new EmptyExpression(), ASSIGNMENT_OPERATOR);
    o->assignment = dynamic_cast<FuncParameter*>(o->assignment->accept(this));
    assert(o->assignment);
  }

  return o;
}

bi::Statement* bi::Resolver::modify(Import* o) {
  o->file->accept(this);
  top()->import(o->file->scope.get());
  return o;
}

bi::Statement* bi::Resolver::modify(VarDeclaration* o) {
  Modifier::modify(o);
  o->param->type->assignable = true;
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

bi::Type* bi::Resolver::modify(RandomType* o) {
  Modifier::modify(o);
  o->left->assignable = true;
  o->right->assignable = true;
  return o;
}

bi::Scope* bi::Resolver::takeMembershipScope() {
  Scope* scope = membershipScope;
  membershipScope = nullptr;
  return scope;
}

bi::Scope* bi::Resolver::top() {
  return scopes.back();
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
