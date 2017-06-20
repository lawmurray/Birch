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

bi::Expression* bi::Resolver::modify(BracketsExpression* o) {
  Modifier::modify(o);

  const int typeSize = o->single->type->count();
  const int indexSize = o->brackets->tupleSize();
  const int rangeDims = o->brackets->tupleDims();
  assert(typeSize == indexSize);  ///@todo Exception

  ArrayType* type = dynamic_cast<ArrayType*>(o->single->type->strip());
  assert(type);
  if (rangeDims > 0) {
    o->type = new ArrayType(type->single->accept(&cloner), rangeDims);
    o->type = o->type->accept(this);
    if (o->single->type->assignable) {
      o->type->accept(&assigner);
    }
  } else {
    o->type = type->single->accept(&cloner)->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(LambdaFunction* o) {
  push();
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->returnType->accept(this);
  o->braces->accept(this);
  o->scope = pop();
  o->type = new FunctionType(o->parens->type->accept(&cloner),
      o->returnType->accept(&cloner));
  o->type->accept(this);

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
  Identifier<Class>* ref =
      dynamic_cast<Identifier<Class>*>(o->left->type->strip());
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
    o->type = new ClassType(type());
    o->type->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Super* o) {
  if (!type()) {
    throw SuperException(o);
  } else if (type()->base->isEmpty()) {
    throw SuperBaseException(o);
  } else {
    Modifier::modify(o);
    o->type = type()->base->accept(&cloner)->accept(this);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Parameter* o) {
  Modifier::modify(o);
  if (!o->name->isEmpty()) {
    top()->add(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<Unknown>* o) {
  return lookup(o)->accept(this);
}

bi::Expression* bi::Resolver::modify(Identifier<Parameter>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<GlobalVariable>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<LocalVariable>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<MemberVariable>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<Function>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<Coroutine>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = new CoroutineType(
      o->target->returnType->accept(&cloner)->accept(this));
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<MemberFunction>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<BinaryOperator>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<UnaryOperator>* o) {
  Scope* memberScope = takeMemberScope();
  Modifier::modify(o);
  resolve(o, memberScope);
  o->type = o->target->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Statement* bi::Resolver::modify(Assignment* o) {
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
    Identifier<Class>* ref =
        dynamic_cast<Identifier<Class>*>(o->left->type->strip());
    if (ref) {
      assert(ref->target);
      memberScope = ref->target->scope.get();
    } else {
      //throw MemberException(o);
    }
    //resolve(o, memberScope);
  }
  if (!o->left->type->assignable) {
    throw NotAssignableException(o->left.get());
  }

  return o;
}

bi::Statement* bi::Resolver::modify(GlobalVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  top()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(LocalVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  top()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(MemberVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  top()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(Function* o) {
  push();
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->returnType = o->returnType->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(Coroutine* o) {
  push();
  ++inInputs;
  o->parens = o->parens->accept(this);
  --inInputs;
  o->returnType = o->returnType->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(Program* o) {
  push();
  o->parens = o->parens->accept(this);
  defer(o->braces.get());
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(MemberFunction* o) {
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

bi::Statement* bi::Resolver::modify(BinaryOperator* o) {
  push();
  ++inInputs;
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  --inInputs;
  o->returnType = o->returnType->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(UnaryOperator* o) {
  push();
  ++inInputs;
  o->single = o->single->accept(this);
  --inInputs;
  o->returnType = o->returnType->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(AssignmentOperator* o) {
  push();
  ++inInputs;
  o->single->type->accept(&assigner);
  --inInputs;
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  //top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(ConversionOperator* o) {
  push();
  o->returnType = o->returnType->accept(this);
  if (!o->braces->isEmpty()) {
    defer(o->braces.get());
  }
  o->scope = pop();
  //top()->add(o);

  return o;
}

bi::Statement* bi::Resolver::modify(Class* o) {
  push();
  o->base = o->base->accept(this);
  o->scope = pop();
  if (!o->base->isEmpty()) {
    ClassType* base = dynamic_cast<ClassType*>(o->base.get());
    assert(base);
    o->scope->inherit(base->target->scope.get());
  }
  top()->add(o);
  push(o->scope.get());
  types.push(o);
  o->braces = o->braces->accept(this);
  types.pop();
  pop();

  ///@todo Check that base type is of class type
  return o;
}

bi::Statement* bi::Resolver::modify(Alias* o) {
  o->base = o->base->accept(this);
  top()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(Basic* o) {
  top()->add(o);
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

bi::Type* bi::Resolver::modify(IdentifierType* o) {
  return lookup(o)->accept(this);
}

bi::Type* bi::Resolver::modify(ClassType* o) {
  Scope* memberScope = takeMemberScope();
  assert(!memberScope);

  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(AliasType* o) {
  Scope* memberScope = takeMemberScope();
  assert(!memberScope);

  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(BasicType* o) {
  Scope* memberScope = takeMemberScope();
  assert(!memberScope);

  Modifier::modify(o);
  resolve(o);
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

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* ref, Scope* scope) {
  LookupResult category = UNRESOLVED;
  if (scope) {
    /* use provided scope, usually a membership scope */
    category = scope->lookup(ref);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(ref);
    }
  }

  /* replace the reference of unknown object type with that of a known one */
  switch (category) {
  case PARAMETER:
    return new Identifier<Parameter>(ref->name, ref->parens.release(),
        ref->loc);
  case GLOBAL_VARIABLE:
    return new Identifier<GlobalVariable>(ref->name, ref->parens.release(),
        ref->loc);
  case LOCAL_VARIABLE:
    return new Identifier<LocalVariable>(ref->name, ref->parens.release(),
        ref->loc);
  case MEMBER_VARIABLE:
    return new Identifier<MemberVariable>(ref->name, ref->parens.release(),
        ref->loc);
  case FUNCTION:
    return new Identifier<Function>(ref->name, ref->parens.release(),
        ref->loc);
  case COROUTINE:
    return new Identifier<Coroutine>(ref->name, ref->parens.release(),
        ref->loc);
  case MEMBER_FUNCTION:
    return new Identifier<MemberFunction>(ref->name, ref->parens.release(),
        ref->loc);
  default:
    throw UnresolvedReferenceException(ref);
  }
}

bi::Type* bi::Resolver::lookup(IdentifierType* ref, Scope* scope) {
  LookupResult category = UNRESOLVED;
  if (scope) {
    /* use provided scope, usually a membership scope */
    category = scope->lookup(ref);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(ref);
    }
  }

  /* replace the reference of unknown object type with that of a known one */
  switch (category) {
  case BASIC:
    return new BasicType(ref->name, ref->loc);
  case CLASS:
    return new ClassType(ref->name, ref->loc);
  case ALIAS:
    return new AliasType(ref->name, ref->loc);
  default:
    throw UnresolvedReferenceException(ref);
  }
}

void bi::Resolver::defer(Statement* o) {
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

bi::Class* bi::Resolver::type() {
  if (types.empty()) {
    return nullptr;
  } else {
    return types.top();
  }
}
