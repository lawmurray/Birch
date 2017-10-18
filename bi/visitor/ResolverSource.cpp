/**
 * @file
 */
#include "bi/visitor/ResolverSource.hpp"

bi::ResolverSource::ResolverSource() :
    currentReturnType(nullptr),
    currentYieldType(nullptr) {
  //
}

bi::ResolverSource::~ResolverSource() {
  //
}

bi::Expression* bi::ResolverSource::modify(Brackets* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Cast* o) {
  Modifier::modify(o);
  if (o->single->type->isClass() ||
      (o->single->type->isOptional() && o->single->type->unwrap()->isClass())) {
    o->type = new OptionalType(o->returnType->accept(&cloner));
    o->type = o->type->accept(this);
    o->type->assignable = false;  // rvalue
    return o;
  } else {
    throw CastException(o);
  }
}

bi::Expression* bi::ResolverSource::modify(Call* o) {
  Modifier::modify(o);
  if (o->single->type->isFunction() || o->single->type->isOverloaded()) {
    o->callType = o->single->type->resolve(o);
    o->type = o->callType->returnType->accept(&cloner)->accept(this);
    o->type->assignable = false;  // rvalue
    return o;
  } else {
    throw NotFunctionException(o);
  }
}

bi::Expression* bi::ResolverSource::modify(BinaryCall* o) {
  auto op = dynamic_cast<OverloadedIdentifier<BinaryOperator>*>(o->single);
  assert(op);
  Modifier::modify(o);
  o->callType = o->single->type->resolve(o);
  o->type = o->callType->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::ResolverSource::modify(UnaryCall* o) {
  Modifier::modify(o);
  o->callType = o->single->type->resolve(o);
  o->type = o->callType->returnType->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::ResolverSource::modify(Slice* o) {
  Modifier::modify(o);

  const int typeSize = o->single->type->count();
  const int sliceSize = o->brackets->tupleSize();
  const int rangeDims = o->brackets->tupleDims();

  if (typeSize != sliceSize) {
    throw SliceException(o, typeSize, sliceSize);
  }

  ArrayType* type = dynamic_cast<ArrayType*>(o->single->type);
  assert(type);
  if (rangeDims > 0) {
    o->type = new ArrayType(type->single->accept(&cloner), rangeDims, o->loc);
    o->type = o->type->accept(this);
  } else {
    o->type = type->single->accept(&cloner)->accept(this);
  }
  if (o->single->type->assignable) {
    o->type->accept(&assigner);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Query* o) {
  Modifier::modify(o);
  if (o->single->type->isFiber() || o->single->type->isOptional()) {
    o->type = new BasicType(new Name("Boolean"), o->loc);
    o->type = o->type->accept(this);
  } else {
    throw QueryException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Get* o) {
  Modifier::modify(o);
  if (o->single->type->isFiber() || o->single->type->isOptional()) {
    o->type = o->single->type->unwrap()->accept(&cloner)->accept(this);
  } else {
    throw GetException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(LambdaFunction* o) {
  scopes.push_back(o->scope);
  o->parens = o->parens = o->parens->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  o->type = new FunctionType(o->parens->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type->accept(this);

  return o;
}

bi::Expression* bi::ResolverSource::modify(Span* o) {
  Modifier::modify(o);
  o->type = o->single->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Index* o) {
  Modifier::modify(o);
  o->type = o->single->type->accept(&cloner)->accept(this);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Range* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Member* o) {
  o->left = o->left->accept(this);
  if (o->left->type->isClass()) {
    memberScope = o->left->type->getClass()->scope;
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type->accept(&cloner)->accept(this);

  return o;
}

bi::Expression* bi::ResolverSource::modify(This* o) {
  if (currentClass) {
    Modifier::modify(o);
    o->type = new ClassType(currentClass, o->loc);
    o->type->accept(this);
  } else {
    throw ThisException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Super* o) {
  if (currentClass) {
    if (currentClass->base->isEmpty()) {
      throw SuperBaseException(o);
    } else {
      Modifier::modify(o);
      o->type = currentClass->base->accept(&cloner);
      o->type->accept(this);
    }
  } else {
    throw SuperException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Nil* o) {
  Modifier::modify(o);
  o->type = new NilType(o->loc);
  o->type->accept(this);
  return o;
}

bi::Expression* bi::ResolverSource::modify(LocalVariable* o) {
  Modifier::modify(o);
  if (!o->parens->isEmpty() || o->value->isEmpty()) {
    o->type->resolveConstructor(o->parens->type);
  }
  o->type->accept(&assigner);
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(MemberParameter* o) {
  o->type->accept(&assigner);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Identifier<Unknown>* o) {
  return lookup(o, memberScope)->accept(this);
}

bi::Expression* bi::ResolverSource::modify(Identifier<Parameter>* o) {
  return modifyVariableIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(Identifier<MemberParameter>* o) {
  return modifyVariableIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(Identifier<GlobalVariable>* o) {
  return modifyVariableIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(Identifier<LocalVariable>* o) {
  return modifyVariableIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(Identifier<MemberVariable>* o) {
  return modifyVariableIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<Function>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(OverloadedIdentifier<Fiber>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<MemberFunction>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<MemberFiber>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<BinaryOperator>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<UnaryOperator>* o) {
  return modifyFunctionIdentifier(o);
}

bi::Statement* bi::ResolverSource::modify(Assignment* o) {
  if (*o->name == "<~") {
    /* replace with equivalent (by definition) code */
    auto right = new Call(
        new Member(o->right->accept(&cloner),
            new Identifier<Unknown>(new Name("tildeLeft"), o->loc), o->loc),
        new Parentheses(new EmptyExpression(o->loc), o->loc), o->loc);
    auto left = o->left->accept(&cloner);
    auto assign = new Assignment(left, new Name("<-"), right, o->loc);
    return assign->accept(this);
  } else if (*o->name == "~>") {
    /* replace with equivalent (by definition) code */
    auto observe = new Call(
        new Member(o->right->accept(&cloner),
            new Identifier<Unknown>(new Name("tildeRight"), o->loc), o->loc),
        new Parentheses(o->left->accept(&cloner), o->loc), o->loc);
    if (currentYieldType) {
      auto yield = new Yield(observe, o->loc);
      return yield->accept(this);
    } else {
      auto stmt = new ExpressionStatement(observe, o->loc);
      return stmt->accept(this);
    }
  } else if (*o->name == "~") {
    /* replace with equivalent (by definition) code */
    auto cond = new Parentheses(
        new Call(
            new Member(o->left->accept(&cloner),
                new Identifier<Unknown>(new Name("isNotMissing"), o->loc),
                o->loc), new Parentheses(new EmptyExpression(o->loc), o->loc),
            o->loc), o->loc);
    auto trueBranch = new Assignment(o->left->accept(&cloner), new Name("~>"),
        o->right->accept(&cloner), o->loc);
    auto falseBranch = new Assignment(o->left->accept(&cloner),
        new Name("<-"), o->right->accept(&cloner), o->loc);
    auto result = new If(cond, trueBranch, falseBranch, o->loc);
    return result->accept(this);
  } else {
    /*
     * An assignment is valid if:
     *
     *   1. the right-side type is a the same as, or a subtype of, the
     *      left-side type (either a polymorphic pointer),
     *   2. a conversion operator for the left-side type is defined in the
     *      right-side (class) type, or
     *   3. an assignment operator for the right-side type is defined in the
     *      left-side (class) type.
     */
    Modifier::modify(o);
    if (!o->left->type->assignable) {
      throw NotAssignableException(o);
    }
    if (!o->right->type->definitely(*o->left->type)) {
      // ^ the first two cases are covered by this check
      Identifier<Class>* ref = dynamic_cast<Identifier<Class>*>(o->left->type);
      if (ref) {
        assert(ref->target);
        memberScope = ref->target->scope;
      } else {
        //throw MemberException(o);
      }
    }
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(GlobalVariable* o) {
  Modifier::modify(o);
  if (!o->parens->isEmpty() || o->value->isEmpty()) {
    o->type->resolveConstructor(o->parens->type);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberVariable* o) {
  Modifier::modify(o);
  if (!o->parens->isEmpty() || o->value->isEmpty()) {
    o->type->resolveConstructor(o->parens->type);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(Function* o) {
  scopes.push_back(o->scope);
  currentReturnType = o->returnType;
  o->braces = o->braces->accept(this);
  currentReturnType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Fiber* o) {
  scopes.push_back(o->scope);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  } else {
    currentYieldType = dynamic_cast<FiberType*>(o->returnType)->single;
  }
  o->braces = o->braces->accept(this);
  currentYieldType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Program* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(&assigner);
  // ^ currently for backwards compatibility of delay_triplet example, can
  //   be updated later
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  currentReturnType = o->returnType;
  o->braces = o->braces->accept(this);
  currentReturnType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  } else {
    currentYieldType = dynamic_cast<FiberType*>(o->returnType)->single;
  }
  o->braces = o->braces->accept(this);
  currentYieldType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(BinaryOperator* o) {
  scopes.push_back(o->scope);
  currentReturnType = o->returnType;
  o->braces = o->braces->accept(this);
  currentReturnType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(UnaryOperator* o) {
  scopes.push_back(o->scope);
  currentReturnType = o->returnType;
  o->braces = o->braces->accept(this);
  currentReturnType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(ConversionOperator* o) {
  scopes.push_back(o->scope);
  currentReturnType = o->returnType;
  o->braces = o->braces->accept(this);
  currentReturnType = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Class* o) {
  scopes.push_back(o->scope);
  currentClass = o;
  o->baseParens = o->baseParens->accept(this);
  o->base->resolveConstructor(o->baseParens->type);
  o->braces = o->braces->accept(this);
  currentClass = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(ExpressionStatement* o) {
  Modifier::modify(o);

  /* when in the body of a fiber and another fiber is called while ignoring
   * its return type, this is syntactic sugar for a loop */
  if (currentYieldType) {
    auto call = dynamic_cast<Call*>(o->single);
    if (call && call->type->isFiber()) {
      auto name = new Name();
      auto var = new LocalVariable(name, o->single->type->accept(&cloner),
          new EmptyExpression(o->loc), o->single, o->loc);
      auto decl = new ExpressionStatement(var, o->loc);
      auto query = new Query(new Identifier<LocalVariable>(name, o->loc),
          o->loc);
      auto get = new Get(new Identifier<LocalVariable>(name, o->loc), o->loc);
      auto yield = new Yield(get, o->loc);
      auto loop = new While(new Parentheses(query, o->loc),
          new Braces(yield, o->loc), o->loc);
      auto result = new List<Statement>(decl, loop, o->loc);

      return result->accept(this);
    }
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(If* o) {
  scopes.push_back(o->scope);
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  scopes.push_back(o->falseScope);
  o->falseBraces = o->falseBraces->accept(this);
  scopes.pop_back();
  checkBoolean(o->cond->strip());
  return o;
}

bi::Statement* bi::ResolverSource::modify(For* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  checkInteger(o->index);
  checkInteger(o->from);
  checkInteger(o->to);
  return o;
}

bi::Statement* bi::ResolverSource::modify(While* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  checkBoolean(o->cond->strip());
  return o;
}

bi::Statement* bi::ResolverSource::modify(Assert* o) {
  Modifier::modify(o);
  checkBoolean(o->cond);
  return o;
}

bi::Statement* bi::ResolverSource::modify(Return* o) {
  Modifier::modify(o);
  if (!currentReturnType) {
    if (!o->single->type->isEmpty()) {
      throw ReturnException(o);
    }
  } else if (!o->single->type->definitely(*currentReturnType)) {
    throw ReturnTypeException(o, currentReturnType);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(Yield* o) {
  Modifier::modify(o);
  if (!currentYieldType) {
    if (!o->single->type->isEmpty()) {
      throw YieldException(o);
    }
  } else if (!o->single->type->definitely(*currentYieldType)) {
    throw YieldTypeException(o, currentYieldType);
  }
  return o;
}
