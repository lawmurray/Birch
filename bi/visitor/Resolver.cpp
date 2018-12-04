/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

bi::Resolver::Resolver(const ResolverStage globalStage) :
    stage(RESOLVER_TYPER),
    globalStage(globalStage),
    annotator(PRIOR_INSTANTIATION) {
  //
}

bi::Resolver::~Resolver() {
  //
}

void bi::Resolver::apply(Package* o) {
  scopes.push_back(o->scope);
  for (stage = RESOLVER_TYPER; stage <= RESOLVER_SOURCE; ++stage) {
    globalStage = stage;
    o->accept(this);
  }
  globalStage = stage;
  scopes.pop_back();
}

bi::Expression* bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type, o->tail->type, o->loc);
  return o;
}

bi::Expression* bi::Resolver::modify(Parentheses* o) {
  Modifier::modify(o);
  if (o->single->width() > 1) {
    o->type = new TupleType(o->single->type, o->loc);
  } else {
    o->type = o->single->type;
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Sequence* o) {
  Modifier::modify(o);
  auto iter = o->single->type->begin();
  if (iter == o->single->type->end()) {
    o->type = new NilType(o->loc);
  } else {
    auto common = *iter;
    ++iter;
    while (common && iter != o->single->type->end()) {
      common = common->common(**iter);
      ++iter;
    }
    if (!common) {
      throw SequenceException(o);
    } else {
      o->type = new SequenceType(common, o->loc);
    }
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Binary* o) {
  Modifier::modify(o);
  o->type = new BinaryType(o->left->type, o->right->type, o->loc);
  return o;
}

bi::Expression* bi::Resolver::modify(Cast* o) {
  Modifier::modify(o);
  if (o->single->type->isPointer()
      || (o->single->type->isOptional()
          && o->single->type->unwrap()->isPointer())) {
    o->type = new OptionalType(o->returnType, o->loc);
    return o;
  } else {
    throw CastException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Call* o) {
  Modifier::modify(o);
  o->callType = o->single->resolve(o);
  o->type = o->callType->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(BinaryCall* o) {
  Modifier::modify(o);
  o->callType = o->single->resolve(o);
  o->type = o->callType->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(UnaryCall* o) {
  Modifier::modify(o);
  o->callType = o->single->resolve(o);
  o->type = o->callType->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Assign* o) {
  if (*o->name == "<~") {
    /* replace with equivalent (by definition) code */
    auto left = o->left;
    auto right = new Call(
        new Member(o->right,
            new Identifier<Unknown>(new Name("simulate"), o->loc), o->loc),
            new EmptyExpression(o->loc), o->loc);
    auto assign = new Assign(left, new Name("<-"), right, o->loc);
    return assign->accept(this);
  } else {
    Modifier::modify(o);
    if (!o->left->isAssignable()) {
      throw NotAssignableException(o);
    }
    if (!o->right->type->definitely(*o->left->type)
        && (!o->left->type->isClass()
            || !o->left->type->getClass()->hasAssignment(o->right->type))) {
      throw AssignmentException(o);
    }
    o->type = o->left->type;
    return o;
  }
}

bi::Expression* bi::Resolver::modify(Slice* o) {
  Modifier::modify(o);

  const int typeDepth = o->single->type->depth();
  const int sliceWidth = o->brackets->width();
  const int rangeDepth = o->brackets->depth();

  ArrayType* type = dynamic_cast<ArrayType*>(o->single->type->canonical());
  if (!type || typeDepth != sliceWidth) {
    throw SliceException(o, typeDepth, sliceWidth);
  }

  if (rangeDepth > 0) {
    o->type = new ArrayType(type->single, rangeDepth, o->loc);
  } else {
    o->type = type->single;
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Query* o) {
  Modifier::modify(o);
  if (o->single->type->isFiber() || o->single->type->isOptional()) {
    o->type = new BasicType(new Name("Boolean"), o->loc);
    o->type = o->type->accept(this);
  } else {
    throw QueryException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Get* o) {
  Modifier::modify(o);
  if (o->single->type->isFiber() || o->single->type->isOptional()) {
    o->type = o->single->type->unwrap();
  } else {
    throw GetException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(LambdaFunction* o) {
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

bi::Expression* bi::Resolver::modify(Span* o) {
  Modifier::modify(o);
  o->type = o->single->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Index* o) {
  Modifier::modify(o);
  o->type = o->single->type;
  checkInteger(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Range* o) {
  Modifier::modify(o);
  checkInteger(o->left);
  checkInteger(o->right);
  return o;
}

bi::Expression* bi::Resolver::modify(Member* o) {
  o->left = o->left->accept(this);
  if (dynamic_cast<Global*>(o->left)) {
    memberScopes.push_back(scopes.front());
  } else if (o->left->type->isClass() && !o->left->type->isWeak()) {
    memberScopes.push_back(o->left->type->getClass()->scope);
  } else {
    throw MemberException(o);
  }
  o->right = o->right->accept(this);
  o->type = o->right->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Super* o) {
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

bi::Expression* bi::Resolver::modify(This* o) {
  if (!classes.empty()) {
    Modifier::modify(o);
    o->type = new PointerType(false, new ClassType(classes.back(), o->loc),
        o->loc);
  } else {
    throw ThisException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Nil* o) {
  Modifier::modify(o);
  o->type = new NilType(o->loc);
  o->type->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(LocalVariable* o) {
  Modifier::modify(o);
  if (o->has(AUTO)) {
    assert(!o->value->isEmpty());
    o->type = o->value->type;
  }
  if (o->needsConstruction()) {
    o->type->resolveConstructor(o);
  }
  if (!o->brackets->isEmpty()) {
    o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
  }
  if (!o->value->isEmpty() && !o->value->type->definitely(*o->type)) {
    throw InitialValueException(o);
  }
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Parameter* o) {
  Modifier::modify(o);
  if (!o->value->isEmpty() && !o->value->type->definitely(*o->type)) {
    throw InitialValueException(o);
  }
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Generic* o) {
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<Unknown>* o) {
  return lookup(o)->accept(this);
}

bi::Expression* bi::Resolver::modify(Identifier<Parameter>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<GlobalVariable>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<LocalVariable>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<MemberVariable>* o) {
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<Unknown>* o) {
  return lookup(o)->accept(this);
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<Function>* o) {
  resolve(o, GLOBAL_SCOPE);
  Modifier::modify(o);
  if (o->target->size() == 1) {
    auto only = instantiate(o, o->target->front());
    o->target = new Overloaded<Function>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<Fiber>* o) {
  resolve(o, GLOBAL_SCOPE);
  Modifier::modify(o);
  if (o->target->size() == 1) {
    auto only = instantiate(o, o->target->front());
    o->target = new Overloaded<Fiber>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<MemberFiber>* o) {
  resolve(o, CLASS_SCOPE);
  Modifier::modify(o);
  if (o->target->size() == 1) {
    auto only = o->target->front();
    o->target = new Overloaded<MemberFiber>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<MemberFunction>* o) {
  resolve(o, CLASS_SCOPE);
  Modifier::modify(o);
  if (o->target->size() == 1) {
    auto only = o->target->front();
    o->target = new Overloaded<MemberFunction>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<BinaryOperator>* o) {
  resolve(o, GLOBAL_SCOPE);
  Modifier::modify(o);
  if (o->target->size() == 1) {
    auto only = o->target->front();
    o->target = new Overloaded<BinaryOperator>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<UnaryOperator>* o) {
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  if (o->target->size() == 1) {
    auto only = o->target->front();
    o->target = new Overloaded<UnaryOperator>(only);
    o->type = new FunctionType(only->params->type, only->returnType);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Assignment* o) {
  /* replace with equivalent (by definition) code */
  if (*o->name == "~>") {
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
  } else {
    assert(*o->name == "~");
    ///@todo Can left be evaluated only once?
    auto cond = new Call(
        new Member(o->left->accept(&cloner),
            new Identifier<Unknown>(new Name("hasValue"), o->loc)),
        new Parentheses(new EmptyExpression(o->loc), o->loc), o->loc);
    auto trueBranch = new Assignment(o->left->accept(&cloner),
        new Name("~>"), o->right->accept(&cloner), o->loc);
    auto falseBranch = new ExpressionStatement(
        new Call(
            new Member(o->left,
                new Identifier<Unknown>(new Name("assume"), o->loc), o->loc),
            o->right, o->loc), o->loc);
    auto result = new If(cond, trueBranch, falseBranch, o->loc);
    return result->accept(this);
  }
}

bi::Statement* bi::Resolver::modify(GlobalVariable* o) {
  if (stage == RESOLVER_HEADER) {
    o->type = o->type->accept(this);
    ///@todo When auto keyword used, type not known here
    if (!o->brackets->isEmpty()) {
      o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
    }
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    o->brackets = o->brackets->accept(this);
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
    if (o->has(AUTO)) {
      assert(!o->value->isEmpty());
      o->type = o->value->type;
    }
    if (o->needsConstruction()) {
      o->type->resolveConstructor(o);
    }
    if (!o->value->isEmpty() && !o->value->type->definitely(*o->type)) {
      throw InitialValueException(o);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(MemberVariable* o) {
  if (stage == RESOLVER_HEADER) {
    o->type = o->type->accept(this);
    ///@todo When auto keyword used, type not known here
    if (!o->brackets->isEmpty()) {
      o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
    }
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(classes.back()->initScope);
    o->brackets = o->brackets->accept(this);
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
    if (o->has(AUTO)) {
      assert(!o->value->isEmpty());
      o->type = o->value->type;
    }
    scopes.pop_back();
    if (o->needsConstruction()) {
      o->type->resolveConstructor(o);
    }
    if (!o->value->isEmpty() && !o->value->type->definitely(*o->type)) {
      throw InitialValueException(o);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Function* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->typeParams = o->typeParams->accept(this);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    if (!o->isInstantiation()) {
      scopes.back()->add(o);
    }
  } else if (stage == RESOLVER_SOURCE && o->isBound()) {
    scopes.push_back(o->scope);
    returnTypes.push_back(o->returnType);
    o->braces = o->braces->accept(this);
    returnTypes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    if (instantiation->stage < stage) {
      instantiation->accept(this);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Fiber* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->typeParams = o->typeParams->accept(this);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    if (!o->isInstantiation()) {
      scopes.back()->add(o);
    }
  } else if (stage == RESOLVER_SOURCE && o->isBound()) {
    scopes.push_back(o->scope);
    yieldTypes.push_back(o->returnType->unwrap());
    o->braces = o->braces->accept(this);
    yieldTypes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    if (instantiation->stage < stage) {
      instantiation->accept(this);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Program* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    scopes.pop_back();
    scopes.back()->add(o);
    ///@todo Check that can assign String to all option types
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(MemberFunction* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    returnTypes.push_back(o->returnType);
    o->braces = o->braces->accept(this);
    returnTypes.pop_back();
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(MemberFiber* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    yieldTypes.push_back(o->returnType->unwrap());
    o->braces = o->braces->accept(this);
    yieldTypes.pop_back();
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(BinaryOperator* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    returnTypes.push_back(o->returnType);
    o->braces = o->braces->accept(this);
    returnTypes.pop_back();
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(UnaryOperator* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    o->returnType = o->returnType->accept(this);
    o->type = new FunctionType(o->params->type, o->returnType, o->loc);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    returnTypes.push_back(o->returnType);
    o->braces = o->braces->accept(this);
    returnTypes.pop_back();
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(AssignmentOperator* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->single = o->single->accept(this);
    scopes.pop_back();
    if (!o->single->type->isValue()) {
      throw AssignmentOperatorException(o);
    }
    classes.back()->addAssignment(o->single->type);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(ConversionOperator* o) {
  if (stage == RESOLVER_SUPER) {
    o->returnType = o->returnType->accept(this);
    if (!o->returnType->isValue()) {
      throw ConversionOperatorException(o);
    }
    classes.back()->addConversion(o->returnType);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    returnTypes.push_back(o->returnType);
    o->braces = o->braces->accept(this);
    returnTypes.pop_back();
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Class* o) {
  if (stage == RESOLVER_TYPER) {
    scopes.push_back(o->scope);
    if (o->base->isEmpty() && o->name->str() != "Object") {
      /* if the class derives from nothing else, then derive from Object,
       * unless this is itself the declaration of the Object class */
      o->base = new ClassType(new Name("Object"), new EmptyType(), o->loc);
    }
    scopes.pop_back();
    if (!o->isInstantiation()) {
      scopes.back()->add(o);
    }
  } else if (stage == RESOLVER_SUPER) {
    scopes.push_back(o->scope);
    classes.push_back(o);
    o->typeParams = o->typeParams->accept(this);
    o->base = o->base->accept(this);
    if (!o->base->isEmpty()) {
      if (!o->base->isClass()) {
        throw BaseException(o);
      }
      o->scope->inherit(o->base->getClass()->scope);
      o->addSuper(o->base);
    }
    o->braces = o->braces->accept(this);  // to visit conversions
    classes.pop_back();
    scopes.pop_back();
  } else if (stage == RESOLVER_HEADER) {
    classes.push_back(o);
    scopes.push_back(o->scope);
    scopes.push_back(o->initScope);
    if (o->isAlias()) {
      o->params = o->base->canonical()->getClass()->params->accept(&cloner);
    }
    o->params = o->params->accept(this);
    scopes.pop_back();
    o->braces = o->braces->accept(this);
    classes.pop_back();
    scopes.pop_back();
  } else if (stage == RESOLVER_SOURCE && o->isBound()) {
    classes.push_back(o);
    scopes.push_back(o->scope);
    scopes.push_back(o->initScope);
    o->args = o->args->accept(this);
    if (!o->alias) {
      o->base->resolveConstructor(o);
    }
    scopes.pop_back();
    o->braces = o->braces->accept(this);
    classes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    if (instantiation->stage < stage) {
      instantiation->accept(this);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Basic* o) {
  if (stage == RESOLVER_TYPER) {
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SUPER) {
    o->base = o->base->accept(this);
    if (!o->base->isEmpty()) {
      if (!o->base->isBasic()) {
        throw BaseException(o);
      }
      o->addSuper(o->base);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(ExpressionStatement* o) {
  Modifier::modify(o);

  /* when in the body of a fiber and another fiber is called while ignoring
   * its return type, this is syntactic sugar for a loop */
  auto call = dynamic_cast<Call*>(o->single);
  if (call && call->type->isFiber()) {
    auto name = new Name();
    auto var = new LocalVariable(NONE, name, o->single->type,
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

bi::Statement* bi::Resolver::modify(If* o) {
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

bi::Statement* bi::Resolver::modify(For* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  checkInteger(o->index);
  checkInteger(o->from);
  checkInteger(o->to);
  return o;
}

bi::Statement* bi::Resolver::modify(While* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  checkBoolean(o->cond->strip());
  return o;
}

bi::Statement* bi::Resolver::modify(DoWhile* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  checkBoolean(o->cond->strip());
  return o;
}

bi::Statement* bi::Resolver::modify(Assert* o) {
  Modifier::modify(o);
  checkBoolean(o->cond);
  return o;
}

bi::Statement* bi::Resolver::modify(Return* o) {
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

bi::Statement* bi::Resolver::modify(Yield* o) {
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

bi::Statement* bi::Resolver::modify(Instantiated<Type>* o) {
  if (stage == RESOLVER_SOURCE) {
    Modifier::modify(o);
    o->single->accept(&annotator);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Instantiated<Expression>* o) {
  if (stage == RESOLVER_SOURCE) {
    Modifier::modify(o);
    o->single->accept(&annotator);
  }
  return o;
}

bi::Type* bi::Resolver::modify(UnknownType* o) {
  return lookup(o)->accept(this);
}

bi::Type* bi::Resolver::modify(ClassType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  o->target = instantiate(o, o->target);
  return o;
}

bi::Type* bi::Resolver::modify(BasicType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  return o;
}

bi::Type* bi::Resolver::modify(GenericType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  return o;
}

bi::Type* bi::Resolver::modify(MemberType* o) {
  o->left = o->left->accept(this);
  if (o->left->isClass()) {
    memberScopes.push_back(o->left->getClass()->scope);
    o->right = o->right->accept(this);
  } else if (o->left->isBound()) {
    throw MemberException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case PARAMETER:
    return new Identifier<Parameter>(o->name, o->loc);
  case GLOBAL_VARIABLE:
    return new Identifier<GlobalVariable>(o->name, o->loc);
  case LOCAL_VARIABLE:
    return new Identifier<LocalVariable>(o->name, o->loc);
  case MEMBER_VARIABLE:
    return new Identifier<MemberVariable>(o->name, o->loc);
  case FUNCTION:
    return new OverloadedIdentifier<Function>(o->name, new EmptyType(o->loc), o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, new EmptyType(o->loc), o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Expression* bi::Resolver::lookup(OverloadedIdentifier<Unknown>* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case FUNCTION:
    return new OverloadedIdentifier<Function>(o->name, o->typeArgs, o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, o->typeArgs, o->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, o->typeArgs, o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, o->typeArgs, o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Type* bi::Resolver::lookup(UnknownType* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case BASIC:
    return new BasicType(o->name, o->loc);
  case CLASS:
    return new PointerType(o->weak,
        new ClassType(o->name, o->typeArgs, o->loc), o->loc);
  case GENERIC:
    return new GenericType(o->name, o->loc);
  default:
    throw UnresolvedException(o);
  }
}

void bi::Resolver::checkBoolean(const Expression* o) {
  static BasicType type(new Name("Boolean"));
  scopes.front()->resolve(&type);
  if (!o->type->definitely(type)) {
    throw ConditionException(o);
  }
}

void bi::Resolver::checkInteger(const Expression* o) {
  static BasicType type(new Name("Integer"));
  scopes.front()->resolve(&type);
  if (!o->type->definitely(type)) {
    throw IndexException(o);
  }
}
