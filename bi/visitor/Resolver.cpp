/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/exception/all.hpp"

#include <sstream>

bi::Resolver::Resolver() :
    membershipScope(nullptr),
    inInputs(false) {
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
  if (!o->left->type->assignable) {
    throw NotAssignableException(o->left.get());
  }
  o->type = o->left->type->accept(&cloner)->accept(this);

  o->backward = new FuncReference(o->left->accept(&cloner), new Name("~>"),
      o->right->accept(&cloner), BINARY_OPERATOR);
  o->backward->accept(this);

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
  resolve(o, membershipScope);
  if (o->dispatcher) {
    o->type = o->dispatcher->type->accept(&cloner)->accept(this);
  } else {
    o->type = o->target->type->accept(&cloner)->accept(this);
  }
  o->type->assignable = false;  // rvalue

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
  ///@todo Check that o->value is of the correct type
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
  models.push(o);
  o->braces = o->braces->accept(this);
  models.pop();
  o->scope = pop();
  top()->add(o);

  if (*o->op != "=") {
    /* create assignment operator */
    Expression* right = new VarParameter(new Name(), new ModelReference(o));
    Expression* left = new VarParameter(new Name(),
        new AssignableType(new ModelReference(o)));
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

  /* initialise with types of first function */
  auto iter = o->funcs.begin();
  FuncParameter* func = *iter;
  Gatherer<VarParameter> gatherer;
  func->parens->accept(&gatherer);
  auto iter2 = gatherer.begin();
  while (iter2 != gatherer.end()) {
    o->paramTypes.push_back((*iter2)->type->accept(&cloner)->accept(this));
    ++iter2;
  }
  o->type = func->result->type->accept(&cloner)->accept(this);
  ++iter;

  /* fill with types of remaining functions */
  while (iter != o->funcs.end()) {
    FuncParameter* func = *iter;
    Gatherer<VarParameter> gatherer;
    func->parens->accept(&gatherer);
    auto iter2 = gatherer.begin();
    auto iter3 = o->paramTypes.begin();
    while (iter2 != gatherer.end() && iter3 != o->paramTypes.end()) {
      *iter3 = combine((*iter2)->type.get(), *iter3);
      ++iter2;
      ++iter3;
    }
    o->type = combine(func->result->type.get(), o->type.get());
    ++iter;
  }

  /* construct parentheses */
  Expression* expr;
  if (o->paramTypes.size() == 0) {
    expr = new EmptyExpression();
  } else {
    int i = 0;
    std::stringstream buf;
    buf << "o" << ++i;
    expr = new VarParameter(new Name(buf.str()), o->paramTypes.back());

    auto iter = o->paramTypes.rbegin();
    while (iter != o->paramTypes.rend()) {
      std::stringstream buf;
      buf << "o" << ++i;
      VarParameter* param = new VarParameter(new Name(buf.str()), *iter);
      expr = new ExpressionList(param, expr);
      ++iter;
    }
  }
  o->parens = new ParenthesesExpression(expr);

  /* normal visit */
  push();
  inInputs = true;
  o->parens = o->parens->accept(this);
  inInputs = false;
  o->type = o->type->accept(this);
  o->scope = pop();

  return o;
}

bi::Type* bi::Resolver::combine(Type* o1, Type* o2) {
  VariantType* variant = dynamic_cast<VariantType*>(o2);
  if (variant) {
    /* this parameter already has a variant type, add the type of the
     * argument to this */
    variant->add(o1->accept(&cloner)->accept(this));
    return variant;
  } else if (*o1 != *o2) {
    /* make a new variant type */
    variant = new VariantType();
    variant->add(o2);
    variant->add(o1->accept(&cloner)->accept(this));
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
    /* sort out dispatchers */
    FuncParameter* param = ref->target;
    Dispatcher* dispatcher = new Dispatcher(param->name, param->mangled);
    dispatcher->insert(param);

    for (auto iter = ref->possibles.begin(); iter != ref->possibles.end();
        ++iter) {
      param = *iter;
      if (*param->mangled == *dispatcher->mangled) {
        /* same pattern as the current dispatcher */
        dispatcher->insert(param);
      } else {
        /* different pattern to the current dispatcher... first finalise the
         * current dispatcher */
        dispatcher->accept(this);
        if (top()->contains(dispatcher)) {
          /* reuse identical dispatcher in the scope */
          Dispatcher* existing = scope->get(dispatcher);
          delete dispatcher;
          dispatcher = existing;
        } else {
          /* add current dispatcher to the scope */
          top()->add(dispatcher);
        }

        /* create new dispatcher for the new pattern */
        dispatcher = new Dispatcher(param->name, param->mangled, dispatcher);
      }
    }
    ref->dispatcher = dispatcher->accept(this);
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
