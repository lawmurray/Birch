/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

bi::Resolver::Resolver(const ResolverStage globalStage) :
    stage(RESOLVER_TYPER),
    globalStage(globalStage),
    annotator(INSTANTIATED),
    inLambda(0),
    inParallel(0),
    inFiber(0),
    inMember(0) {
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
    } else if (common->isArray()) {
      o->type = new ArrayType(common->element(), common->depth() + 1, o->loc);
    } else {
      o->type = new ArrayType(common, 1, o->loc);
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
  o->type = new OptionalType(o->returnType, o->loc);
  return o;
}

bi::Expression* bi::Resolver::modify(Call<Unknown>* o) {
  Modifier::modify(o);
  return lookup(o)->accept(this);
}

bi::Expression* bi::Resolver::modify(Call<Parameter>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<LocalVariable>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<MemberVariable>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<GlobalVariable>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<Function>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<MemberFunction>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<Fiber>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<MemberFiber>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<UnaryOperator>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Call<BinaryOperator>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = dynamic_cast<FunctionType*>(o->target->type)->returnType;
  return o;
}

bi::Expression* bi::Resolver::modify(Assign* o) {
  Modifier::modify(o);
  if (!o->right->type->isAssignable(*o->left->type)) {
    throw AssignmentException(o);
  }
  if (!o->left->isAssignable()) {
    /* use of an explicitly-declared assignment operator to assign a value of
     * basic type to an object of class type is okay here, otherwise not */
    if (!o->left->type->isClass() || o->right->type->isClass()) {
      throw NotAssignableException(o);
    }
  }
  o->type = o->left->type;
  return o;
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
  if (o->single->type->isFiber() || o->single->type->isOptional() ||
      o->single->type->isWeak()) {
    o->type = new BasicType(new Name("Boolean"), o->loc);
    o->type = o->type->accept(this);
  } else {
    throw QueryException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Get* o) {
  Modifier::modify(o);
  if (o->single->type->isFiber() || o->single->type->isOptional() ||
      o->single->type->isWeak()) {
    o->type = o->single->type->unwrap();
  } else {
    throw GetException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(LambdaFunction* o) {
  ++inLambda;
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  returnTypes.push_back(o->returnType);
  o->braces = o->braces->accept(this);
  returnTypes.pop_back();
  scopes.pop_back();
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  --inLambda;
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
  if (o->left->type->isClass() && !o->left->type->isWeak()) {
    memberScopes.push_back(o->left->type->getClass()->scope);
  } else {
    throw MemberException(o);
  }
  ++inMember;
  o->right = o->right->accept(this);
  --inMember;
  o->type = o->right->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Global* o) {
  memberScopes.push_back(scopes.front());
  o->single = o->single->accept(this);
  o->type = o->single->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Super* o) {
  if (!classes.empty()) {
    if (classes.back()->base->isEmpty()) {
      throw SuperBaseException(o);
    } else {
      Modifier::modify(o);
      o->type = classes.back()->base;
    }
  } else {
    throw SuperException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(This* o) {
  if (!classes.empty()) {
    Modifier::modify(o);
    o->type = new ClassType(classes.back(), o->loc);
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

bi::Expression* bi::Resolver::modify(Parameter* o) {
  Modifier::modify(o);
  if (!o->value->isEmpty() && !(o->value->type->isConvertible(*o->type) ||
      o->value->type->isConvertible(*o->type->element()))) {
    throw InitialValueException(o);
  }
  if (inFiber && !inLambda && !inParallel) {
    o->set(IN_FIBER);
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

bi::Expression* bi::Resolver::modify(Identifier<MemberVariable>* o) {
  if (!inMember) {
    return (new Member(new This(o->loc), o, o->loc))->accept(this);
  } else {
    Modifier::modify(o);
    resolve(o, CLASS_SCOPE);
    o->type = o->target->type;
    return o;
  }
}

bi::Expression* bi::Resolver::modify(Identifier<LocalVariable>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<ParallelVariable>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
  o->type = o->target->type;
  return o;
}

bi::Expression* bi::Resolver::modify(Identifier<ForVariable>* o) {
  Modifier::modify(o);
  resolve(o, LOCAL_SCOPE);
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
  } else {
    auto target = new Overloaded<Function>();
    for (auto overload : *o->target) {
      if (overload->isGeneric() == !o->typeArgs->isEmpty()) {
        target->add(instantiate(o, overload));
      }
    }
    o->target = target;
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
  } else {
    auto target = new Overloaded<Fiber>();
    for (auto overload : *o->target) {
      if (overload->isGeneric() == !o->typeArgs->isEmpty()) {
        target->add(instantiate(o, overload));
      }
    }
    o->target = target;
  }
  return o;
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<MemberFunction>* o) {
  if (!inMember) {
    return (new Member(new This(o->loc), o, o->loc))->accept(this);
  } else {
    resolve(o, CLASS_SCOPE);
    Modifier::modify(o);
    if (o->target->size() == 1) {
      auto only = o->target->front();
      o->target = new Overloaded<MemberFunction>(only);
      o->type = new FunctionType(only->params->type, only->returnType);
    }
    return o;
  }
}

bi::Expression* bi::Resolver::modify(OverloadedIdentifier<MemberFiber>* o) {
  if (!inMember) {
    return (new Member(new This(o->loc), o, o->loc))->accept(this);
  } else {
    resolve(o, CLASS_SCOPE);
    Modifier::modify(o);
    if (o->target->size() == 1) {
      auto only = o->target->front();
      o->target = new Overloaded<MemberFiber>(only);
      o->type = new FunctionType(only->params->type, only->returnType);
    }
    return o;
  }
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

bi::Statement* bi::Resolver::modify(Assume* o) {
  Statement* result = nullptr;
  if (*o->name == "<-?") {
    auto tmp = new LocalVariable(o->right, o->loc);
    auto ref = new Identifier<Unknown>(tmp->name, o->loc);
    auto cond = new Query(ref->accept(&cloner), o->loc);
    auto trueBranch = new ExpressionStatement(
        new Assign(o->left, new Name("<-"),
            new Get(ref->accept(&cloner), o->loc), o->loc), o->loc);
    auto falseBranch = new EmptyStatement(o->loc);
    auto conditional = new If(cond, trueBranch, falseBranch, o->loc);
    result = new StatementList(tmp, conditional, o->loc);
  } else {
    o->right = o->right->accept(this);
    auto valueType = getValueType(o->right->type);
    if (!valueType) {
      throw AssumeException(o);
    } else {
      valueType = valueType->accept(&cloner);
    }
    if (*o->name == "<~") {
      auto identifier = new OverloadedIdentifier<Unknown>(
          new Name("SimulateEvent"), valueType, o->loc);
      auto call = new Call<Unknown>(identifier, o->right->accept(&cloner));
      auto tmp = new LocalVariable(call, o->loc);
      auto yield = new Yield(new Identifier<Unknown>(tmp->name, o->loc),
          o->loc);
      auto member = new Member(new Identifier<Unknown>(tmp->name, o->loc),
          new OverloadedIdentifier<Unknown>(new Name("value"), new EmptyType(),
              o->loc), o->loc);
      auto value = new Call<Unknown>(member, new EmptyExpression(), o->loc);
      auto assign = new ExpressionStatement(new Assign(o->left,
          new Name("<-"), value, o->loc), o->loc);
      result = new StatementList(tmp, new StatementList(yield, assign,
          o->loc), o->loc);
    } else if (*o->name == "~>") {
      auto identifier = new OverloadedIdentifier<Unknown>(
          new Name("ObserveEvent"), valueType, o->loc);
      auto args = new ExpressionList(o->left, o->right->accept(&cloner),
          o->loc);
      result = new Yield(new Call<Unknown>(identifier, args, o->loc), o->loc);
    } else if (*o->name == "~") {
      auto identifier = new OverloadedIdentifier<Unknown>(
          new Name("AssumeEvent"), valueType, o->loc);
      auto args = new ExpressionList(o->left, o->right->accept(&cloner),
          o->loc);
      result = new Yield(new Call<Unknown>(identifier, args, o->loc), o->loc);
    } else {
      assert(false);
    }
  }
  return result->accept(this);
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
    for (auto iter : *o->brackets) {
      checkInteger(iter);
    }
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
    if (o->has(AUTO)) {
      assert(!o->value->isEmpty());
      if (!o->value->type->isEmpty()) {
        o->type = o->value->type;
      } else {
        throw InitialValueException(o);
      }
    }
    if (o->needsConstruction()) {
      o->type->resolveConstructor(o);
    }
    if (!o->value->isEmpty() && !(o->value->type->isConvertible(*o->type) ||
        o->value->type->isConvertible(*o->type->element()))) {
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
    for (auto iter : *o->brackets) {
      checkInteger(iter);
    }
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
    if (o->has(AUTO)) {
      assert(!o->value->isEmpty());
      if (!o->value->type->isEmpty()) {
        o->type = o->value->type;
      } else {
        throw InitialValueException(o);
      }
    }
    scopes.pop_back();
    if (o->needsConstruction()) {
      o->type->resolveConstructor(o);
    }
    if (!o->value->isEmpty() && !(o->value->type->isConvertible(*o->type) ||
        o->value->type->isConvertible(*o->type->element()))) {
      throw InitialValueException(o);
    }
  }
  return o;
}

bi::Statement* bi::Resolver::modify(LocalVariable* o) {
  Modifier::modify(o);
  if (o->has(AUTO)) {
    assert(!o->value->isEmpty());
    if (!o->value->type->isEmpty()) {
      o->type = o->value->type;
    } else {
      throw InitialValueException(o);
    }
  }
  if (o->needsConstruction()) {
    o->type->resolveConstructor(o);
  }
  if (!o->brackets->isEmpty()) {
    o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
  }
  for (auto iter : *o->brackets) {
    checkInteger(iter);
  }
  if (!o->value->isEmpty() && !(o->value->type->isConvertible(*o->type) ||
      o->value->type->isConvertible(*o->type->element()))) {
    throw InitialValueException(o);
  }
  if (inFiber && !inLambda && !inParallel) {
    o->set(IN_FIBER);
  }
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(ForVariable* o) {
  Modifier::modify(o);
  if (inFiber && !inLambda && !inParallel) {
    o->set(IN_FIBER);
  }
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::Resolver::modify(ParallelVariable* o) {
  Modifier::modify(o);
  if (inFiber && !inLambda && !inParallel) {
    o->set(IN_FIBER);
  }
  scopes.back()->add(o);
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
  ++inFiber;
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
  --inFiber;

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
  if (o->has(ABSTRACT) && !o->braces->isEmpty()) {
    throw AbstractBodyException(o);
  }
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
  if (o->has(ABSTRACT) && !o->braces->isEmpty()) {
    throw AbstractBodyException(o);
  }
  ++inFiber;
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
  --inFiber;
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
      o->base = new ClassType(false, new Name("Object"), new EmptyType(), o->loc);
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
    if (o->isBound() && !o->base->isEmpty()) {
      if (!o->base->isClass()) {
        throw BaseException(o);
      } else if (!o->isAlias() && o->base->getClass()->has(FINAL)) {
        throw FinalException(o);
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
  auto fiberCall = dynamic_cast<Call<Fiber>*>(o->single);
  auto memberFiberCall = dynamic_cast<Call<MemberFiber>*>(o->single);
  if (fiberCall || memberFiberCall) {
    auto name = new Name();
    auto var = new LocalVariable(AUTO, name, new EmptyType(o->loc),
        new EmptyExpression(o->loc), new EmptyExpression(o->loc),
        o->single->accept(&cloner), o->loc);
    auto query = new Query(new Identifier<Unknown>(name, o->loc),
        o->loc);
    auto get = new Get(new Identifier<Unknown>(name, o->loc), o->loc);
    auto yield = new Yield(get, o->loc);
    auto loop = new While(new Parentheses(query, o->loc),
        new Braces(yield, o->loc), o->loc);
    auto result = new StatementList(var, loop, o->loc);

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
  checkInteger(o->from);
  checkInteger(o->to);
  return o;
}

bi::Statement* bi::Resolver::modify(Parallel* o) {
  ++inParallel;
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  --inParallel;
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
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  o->cond = o->cond->accept(this);
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
  } else if (!o->single->type->isConvertible(*returnTypes.back())) {
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
  } else if (!o->single->type->isConvertible(*yieldTypes.back())) {
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
  } else {
    throw MemberException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* o) {
  Lookup category = UNRESOLVED;
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
  case MEMBER_VARIABLE:
    return new Identifier<MemberVariable>(o->name, o->loc);
  case LOCAL_VARIABLE:
    return new Identifier<LocalVariable>(o->name, o->loc);
  case FOR_VARIABLE:
    return new Identifier<ForVariable>(o->name, o->loc);
  case PARALLEL_VARIABLE:
    return new Identifier<ParallelVariable>(o->name, o->loc);
  case FUNCTION:
    return new OverloadedIdentifier<Function>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, new EmptyType(o->loc), o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, new EmptyType(o->loc), o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Expression* bi::Resolver::lookup(OverloadedIdentifier<Unknown>* o) {
  Lookup category = UNRESOLVED;
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
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, o->typeArgs, o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, o->typeArgs, o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, o->typeArgs, o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Type* bi::Resolver::lookup(UnknownType* o) {
  Lookup category = UNRESOLVED;
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
  if (category == BASIC) {
    Type* type = new BasicType(o->name, o->loc);
    if (o->weak) {
      throw WeakException(o);
    }
    return type;
  } else if (category == CLASS || (category == GENERIC && o->weak)) {
    /* a generic annotated weak must be for a class type */
    return new ClassType(o->weak, o->name, o->typeArgs, o->loc);
  } else if (category == GENERIC) {
    return new GenericType(o->name, o->loc);
  } else {
    throw UnresolvedException(o);
  }
}

bi::Expression* bi::Resolver::lookup(Call<Unknown>* o) {
  auto category = o->single->lookup(o->args);
  auto single = o->single->accept(&cloner);
  auto args = o->args->accept(&cloner);
  switch (category) {
  case PARAMETER:
    return new Call<Parameter>(single, args, o->loc);
  case LOCAL_VARIABLE:
    return new Call<LocalVariable>(single, args, o->loc);
  case MEMBER_VARIABLE:
    return new Call<MemberVariable>(single, args, o->loc);
  case GLOBAL_VARIABLE:
    return new Call<GlobalVariable>(single, args, o->loc);
  case FUNCTION:
    return new Call<Function>(single, args, o->loc);
  case MEMBER_FUNCTION:
    return new Call<MemberFunction>(single, args, o->loc);
  case FIBER:
    return new Call<Fiber>(single, args, o->loc);
  case MEMBER_FIBER:
    return new Call<MemberFiber>(single, args, o->loc);
  case UNARY_OPERATOR:
    return new Call<UnaryOperator>(single, args, o->loc);
  case BINARY_OPERATOR:
    return new Call<BinaryOperator>(single, args, o->loc);
  default:
    assert(false);
  }
}

void bi::Resolver::checkBoolean(const Expression* o) {
  static BasicType type(new Name("Boolean"));
  scopes.front()->resolve(&type);
  if (!o->type->isConvertible(type)) {
    throw ConditionException(o);
  }
}

void bi::Resolver::checkInteger(const Expression* o) {
  static BasicType type(new Name("Integer"));
  scopes.front()->resolve(&type);
  if (!o->type->isConvertible(type)) {
    throw IndexException(o);
  }
}

bi::Type* bi::Resolver::getValueType(const Type* o) {
  auto type = dynamic_cast<const ClassType*>(o->canonical());
  while (type && type->name->str() != "Distribution") {
    type = dynamic_cast<const ClassType*>(type->getClass()->base->canonical());
  }
  if (!type || type->isEmpty()) {
    return nullptr;
  } else {
    return type->typeArgs->canonical();
  }
}
