/**
 * @file
 */
#include "bi/visitor/ResolverSource.hpp"

bi::ResolverSource::ResolverSource() {
  //
}

bi::ResolverSource::~ResolverSource() {
  //
}

bi::Expression* bi::ResolverSource::modify(Brackets* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::ResolverSource::modify(Call* o) {
  Modifier::modify(o);
  if (o->single->type->isFunction() || o->single->type->isOverloaded()) {
    o->type = o->single->type->resolve(o->args->type);
    o->type = o->type->accept(&cloner)->accept(this);
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
  o->type = o->single->type->resolve(o->args->type);
  o->type = o->type->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::ResolverSource::modify(UnaryCall* o) {
  Modifier::modify(o);
  o->type = o->single->type->resolve(o->args->type);
  o->type = o->type->accept(&cloner)->accept(this);
  o->type->assignable = false;  // rvalue
  return o;
}

bi::Expression* bi::ResolverSource::modify(Slice* o) {
  Modifier::modify(o);

  const int typeSize = o->single->type->count();
  const int indexSize = o->brackets->tupleSize();
  const int rangeDims = o->brackets->tupleDims();
  assert(typeSize == indexSize);  ///@todo Exception

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
  o->parens = o->parens->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
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
  ClassType* type = dynamic_cast<ClassType*>(o->left->type);
  if (type) {
    assert(type->target);
    memberScope = type->target->scope;
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type->accept(&cloner)->accept(this);

  return o;
}

bi::Expression* bi::ResolverSource::modify(This* o) {
  if (classes.size() > 0) {
    Modifier::modify(o);
    o->type = new ClassType(classes.top(), o->loc);
    o->type->accept(this);
  } else {
    throw ThisException(o);
  }
  return o;
}

bi::Expression* bi::ResolverSource::modify(Super* o) {
  if (classes.size() > 0) {
    if (classes.top()->base->isEmpty()) {
      throw SuperBaseException(o);
    } else {
      Modifier::modify(o);
      o->type = classes.top()->base->accept(&cloner);
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
    auto initialize = new Assignment(o->left->accept(&cloner), new Name("~"),
        o->right->accept(&cloner));
    auto value = new Call(new Identifier<Unknown>(new Name("value")),
        new Parentheses());
    auto call = new ExpressionStatement(
        new Member(o->left->accept(&cloner), value));
    auto result = new List<Statement>(initialize, call, o->loc);
    return result->accept(this);
  } else if (*o->name == "~>") {
    /* replace with equivalent (by definition) code */
    auto result = new Assignment(o->left->accept(&cloner), new Name("~"),
        o->right->accept(&cloner));
    return result->accept(this);
  } else if (*o->name == "~") {
    Modifier::modify(o);
    if (!o->left->type->assignable) {
      throw NotAssignableException(o);
    }
    ///@todo Check that both sides are of Delay type
    return o;
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

bi::Statement* bi::ResolverSource::modify(LocalVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverSource::modify(Function* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Fiber* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Program* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(BinaryOperator* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(UnaryOperator* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(ConversionOperator* o) {
  scopes.push_back(o->scope);
  o->braces->accept(this);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSource::modify(Class* o) {
  scopes.push_back(o->scope);
  classes.push(o);
  o->baseParens = o->baseParens->accept(this);
  o->braces = o->braces->accept(this);
  classes.pop();
  scopes.pop_back();
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
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::ResolverSource::modify(For* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  ///@todo Check that index, from and to are of type Integer
  return o;
}

bi::Statement* bi::ResolverSource::modify(While* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::ResolverSource::modify(Assert* o) {
  Modifier::modify(o);
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::ResolverSource::modify(Return* o) {
  Modifier::modify(o);
  ///@todo Check that the type of the expression is correct
  return o;
}

bi::Statement* bi::ResolverSource::modify(Yield* o) {
  Modifier::modify(o);
  ///@todo Check that the type of the expression is correct
  return o;
}
