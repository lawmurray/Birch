/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

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

bi::Expression* bi::Resolver::modify(List<Expression>* o) {
  Modifier::modify(o);
  o->type = new List<Type>(o->head->type->accept(&cloner),
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

bi::Expression* bi::Resolver::modify(Span* o) {
  Modifier::modify(o);
  o->type = o->single->type->accept(&cloner)->accept(this);
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

  return o;
}

bi::Expression* bi::Resolver::modify(This* o) {
  if (!type()) {
    throw ThisException(o);
  } else {
    Modifier::modify(o);
    o->type = new TypeReference(type());
    o->type->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Super* o) {
  if (!type()) {
    throw SuperException(o);
  } else if (!type()->super()) {
    throw SuperBaseException(o);
  } else {
    Modifier::modify(o);
    o->type = new TypeReference(type()->super());
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
      o->type->accept(&assigner);
    }
  } else {
    o->type = type->single->accept(&cloner)->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(VarReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  if (!o->name->isEmpty()) {
    resolve(o, memberScope);
    o->form = o->target->form;
    o->type = o->target->type->accept(&cloner)->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(FuncReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->form = o->target->form;
  if (o->target->isCoroutine() && !o->target->isLambda()) {
    o->type = new CoroutineType(
        o->target->type->accept(&cloner)->accept(this));
  } else {
    o->type = o->target->type->accept(&cloner)->accept(this);
  }
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(BinaryReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(UnaryReference* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Statement* bi::Resolver::modify(AssignmentReference* o) {
  /*
   * Use of an assignment operator is valid if:
   *
   *   1. the right-side type is a the same as, or a subtype of, the
   *      left-side type (either a polymorphic pointer),
   *   2. a conversion operator for the left-side type is defined in the
   *      right-side (class) type, or
   *   3. an assignment operator for the right-side type is defined in the
   *      left-side (class) type.
   */
  Modifier::modify(o);
  if (!o->right->type->definitely(*o->left->type)) {
    // ^ the first two cases are covered by this check
    TypeReference* ref = dynamic_cast<TypeReference*>(o->left->type->strip());
    if (ref) {
      assert(ref->target);
      memberScope = ref->target->scope.get();
    } else {
      //throw MemberException(o);
    }
    resolve(o, memberScope);
  }
  if (!o->left->type->assignable) {
    throw NotAssignableException(o->left.get());
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
  if (!o->value->isEmpty()) {
    if (!o->type->assignable) {
      throw NotAssignableException(o);
    } else if (!o->value->type->definitely(*o->type)) {
      throw InvalidAssignmentException(o);
    }
  }
  ///@todo Check constructor arguments
  ///@todo Check assignment operator for value
  return o;
}

bi::Statement* bi::Resolver::modify(FuncParameter* o) {
  push();
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(BinaryParameter* o) {
  push();
  ++inInputs;
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  --inInputs;
  o->type = o->type->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(UnaryParameter* o) {
  push();
  ++inInputs;
  o->single = o->single->accept(this);
  --inInputs;
  o->type = o->type->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(AssignmentParameter* o) {
  push();
  ++inInputs;
  o->single->type->accept(&assigner);
  --inInputs;
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(ConversionParameter* o) {
  push();
  o->type = o->type->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens = o->parens->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Type* bi::Resolver::modify(TypeParameter* o) {
  push();
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->base = o->base->accept(this);
  o->baseParens = o->baseParens->accept(this);
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

  ///@todo Check that the type and its base are both struct or both class
  ///@todo Check base type constructor arguments
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
