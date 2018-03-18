/**
 * @file
 */
#include "bi/visitor/ResolverSource.hpp"

#include "bi/visitor/ResolverSuper.hpp"
#include "bi/visitor/ResolverHeader.hpp"

bi::ResolverSource::ResolverSource(Scope* rootScope) :
    Resolver(rootScope, true) {
  //
}

bi::ResolverSource::~ResolverSource() {
  //
}

bi::Expression* bi::ResolverSource::modify(Cast* o) {
  Modifier::modify(o);
  if (o->single->type->isPointer()
      || (o->single->type->isOptional()
          && o->single->type->unwrap()->isPointer())) {
    o->type = new OptionalType(o->returnType, o->loc);
    return o;
  } else {
    throw CastException(o);
  }
}

bi::Expression* bi::ResolverSource::modify(Call* o) {
  Modifier::modify(o);
  if (o->single->type->isFunction() || o->single->type->isOverloaded()) {
    o->callType = o->single->type->resolve(o);
    o->type = o->callType->returnType;
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
  o->type = o->callType->returnType;
  return o;
}

bi::Expression* bi::ResolverSource::modify(UnaryCall* o) {
  Modifier::modify(o);
  o->callType = o->single->type->resolve(o);
  o->type = o->callType->returnType;
  return o;
}

bi::Expression* bi::ResolverSource::modify(Slice* o) {
  Modifier::modify(o);

  const int typeDepth = o->single->type->depth();
  const int sliceWidth = o->brackets->width();
  const int rangeDepth = o->brackets->depth();

  if (typeDepth != sliceWidth) {
    throw SliceException(o, typeDepth, sliceWidth);
  }

  ArrayType* type = dynamic_cast<ArrayType*>(o->single->type->canonical());
  assert(type);
  if (rangeDepth > 0) {
    o->type = new ArrayType(type->single, rangeDepth, o->loc);
  } else {
    o->type = type->single;
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
    o->type = o->single->type->unwrap();
  } else {
    throw GetException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(LambdaFunction* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);

  return o;
}

bi::Expression* bi::ResolverSource::modify(Span* o) {
  Modifier::modify(o);
  o->type = o->single->type;
  return o;
}

bi::Expression* bi::ResolverSource::modify(Index* o) {
  Modifier::modify(o);
  o->type = o->single->type;
  checkInteger(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Range* o) {
  Modifier::modify(o);
  checkInteger(o->left);
  checkInteger(o->right);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Member* o) {
  o->left = o->left->accept(this);
  if (o->left->type->isClass()) {
    memberScopes.push_back(o->left->type->getClass()->scope);
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type;

  return o;
}

bi::Expression* bi::ResolverSource::modify(This* o) {
  if (!classes.empty()) {
    Modifier::modify(o);
    o->type = new PointerType(false, new ClassType(classes.back(), o->loc),
        o->loc);
  } else {
    throw ThisException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Super* o) {
  if (!classes.empty()) {
    if (classes.back()->base->isEmpty()) {
      throw SuperBaseException(o);
    } else {
      Modifier::modify(o);
      o->type = new PointerType(false, classes.back()->base, o->loc);
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

bi::Expression* bi::ResolverSource::modify(Parameter* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Generic* o) {
  return o;
}

bi::Expression* bi::ResolverSource::modify(LocalVariable* o) {
  Modifier::modify(o);
  if (o->needsConstruction()) {
    o->type->resolveConstructor(o);
  }
  if (!o->brackets->isEmpty()) {
    o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
  }
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Identifier<Unknown>* o) {
  return lookup(o)->accept(this);
}

bi::Expression* bi::ResolverSource::modify(Identifier<Parameter>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::ResolverSource::modify(Identifier<GlobalVariable>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::ResolverSource::modify(Identifier<LocalVariable>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::ResolverSource::modify(Identifier<MemberVariable>* o) {
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<Function>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Expression* bi::ResolverSource::modify(OverloadedIdentifier<Fiber>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<MemberFunction>* o) {
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<MemberFiber>* o) {
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<BinaryOperator>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Expression* bi::ResolverSource::modify(
    OverloadedIdentifier<UnaryOperator>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}

bi::Statement* bi::ResolverSource::modify(Assignment* o) {
  if (*o->name == "<~") {
    /* replace with equivalent (by definition) code */
    auto left = o->left;
    auto right = new Call(
        new Member(o->right,
            new Identifier<Unknown>(new Name("simulate"), o->loc), o->loc),
        new EmptyExpression(o->loc), o->loc);
    auto assign = new Assignment(left, new Name("<-"), right, o->loc);
    return assign->accept(this);
  } else if (*o->name == "~>") {
    /* replace with equivalent (by definition) code */
    auto observe = new Call(
        new Member(o->right,
            new Identifier<Unknown>(new Name("observe"), o->loc), o->loc),
        o->left, o->loc);
    if (!yieldTypes.empty()) {
      auto yield = new Yield(observe, o->loc);
      return yield->accept(this);
    } else {
      auto stmt = new ExpressionStatement(observe, o->loc);
      return stmt->accept(this);
    }
  } else if (*o->name == "~") {
    /* replace with equivalent (by definition) code */
    ///@todo Can left be evaluated only once?
    auto cond = new Parentheses(
        new Call(
            new Member(o->left->accept(&cloner),
                new Identifier<Unknown>(new Name("isMissing"), o->loc)),
            new Parentheses(new EmptyExpression(o->loc), o->loc), o->loc),
        o->loc);
    auto trueBranch = new Assignment(o->left->accept(&cloner), new Name("<-"),
        o->right->accept(&cloner), o->loc);
    auto falseBranch = new Assignment(o->left->accept(&cloner),
        new Name("~>"), o->right->accept(&cloner), o->loc);
    auto result = new If(cond, trueBranch, falseBranch, o->loc);
    return result->accept(this);
  } else {
    /* assignment operator */
    Modifier::modify(o);
    if (!o->left->isAssignable()) {
      throw NotAssignableException(o);
    }
    if (!o->right->type->definitely(*o->left->type)
        && (!o->left->type->isClass()
            || !o->left->type->getClass()->hasAssignment(o->right->type))) {
      throw AssignmentException(o);
    }
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(GlobalVariable* o) {
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  if (o->needsConstruction()) {
    o->type->resolveConstructor(o);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(Function* o) {
  scopes.push_back(o->scope);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Fiber* o) {
  scopes.push_back(o->scope);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  } else {
    yieldTypes.push_back(dynamic_cast<FiberType*>(o->returnType)->single);
  }
  o->braces = o->braces->accept(this);
  yieldTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Program* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(BinaryOperator* o) {
  scopes.push_back(o->scope);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(UnaryOperator* o) {
  scopes.push_back(o->scope);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberVariable* o) {
  scopes.push_back(classes.back()->initScope);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  scopes.pop_back();
  if (o->needsConstruction()) {
    o->type->resolveConstructor(o);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  } else {
    yieldTypes.push_back(dynamic_cast<FiberType*>(o->returnType)->single);
  }
  o->braces = o->braces->accept(this);
  yieldTypes.pop_back();
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
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Class* o) {
  if (o->state < RESOLVED_SUPER) {
    ResolverSuper resolver(scopes.front());
    o->accept(&resolver);
  }
  if (o->state < RESOLVED_HEADER) {
    ResolverHeader resolver(scopes.front());
    o->accept(&resolver);
  }
  if (o->state < RESOLVED_SOURCE) {
    classes.push_back(o);
    scopes.push_back(o->scope);
    scopes.push_back(o->initScope);
    o->args = o->args->accept(this);
    if (!o->alias) {
      o->base->resolveConstructor(o);
    }
    scopes.pop_back();
    o->braces = o->braces->accept(this);
    o->state = RESOLVED_SOURCE;
    classes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(Basic* o) {
  return o;
}

bi::Statement* bi::ResolverSource::modify(Explicit* o) {
  return o;
}

bi::Statement* bi::ResolverSource::modify(ExpressionStatement* o) {
  Modifier::modify(o);

  /* when in the body of a fiber and another fiber is called while ignoring
   * its return type, this is syntactic sugar for a loop */
  auto call = dynamic_cast<Call*>(o->single);
  if (call && call->type->isFiber()) {
    auto name = new Name();
    auto var = new LocalVariable(name, o->single->type->accept(&cloner),
        new EmptyExpression(o->loc), new EmptyExpression(o->loc),
        o->single->accept(&cloner), o->loc);
    auto decl = new ExpressionStatement(var, o->loc);
    auto query = new Query(new Identifier<LocalVariable>(name, o->loc),
        o->loc);
    auto get = new Get(new Identifier<LocalVariable>(name, o->loc), o->loc);
    auto yield = new Yield(get, o->loc);
    auto loop = new While(new Parentheses(query, o->loc),
        new Braces(yield, o->loc), o->loc);
    auto result = new StatementList(decl, loop, o->loc);

    return result->accept(this);
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

bi::Statement* bi::ResolverSource::modify(DoWhile* o) {
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
  if (returnTypes.empty()) {
    if (!o->single->type->isEmpty()) {
      throw ReturnException(o);
    }
  } else if (!o->single->type->definitely(*returnTypes.back())) {
    throw ReturnTypeException(o, returnTypes.back());
  }
  return o;
}

bi::Statement* bi::ResolverSource::modify(Yield* o) {
  Modifier::modify(o);
  if (yieldTypes.empty()) {
    if (!o->single->type->isEmpty()) {
      throw YieldException(o);
    }
  } else if (!o->single->type->definitely(*yieldTypes.back())) {
    throw YieldTypeException(o, yieldTypes.back());
  }
  return o;
}
