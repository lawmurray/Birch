/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

bi::Resolver::Resolver(const ResolverStage globalStage) :
    stage(RESOLVER_HEADER),
    globalStage(globalStage),
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
  for (stage = RESOLVER_HEADER; stage < RESOLVER_FINISHED; ++stage) {
    globalStage = stage;
    o->accept(this);
  }
  globalStage = stage;
  scopes.pop_back();
}

bi::Expression* bi::Resolver::modify(Assign* o) {
  Modifier::modify(o);
  if (!o->left->isAssignable()) {
    /* use of an explicitly-declared assignment operator to assign a value of
     * basic type to an object of class type is okay here, otherwise not */
    if (!o->left->type->isClass() || o->right->type->isClass()) {
      throw NotAssignableException(o);
    }
  }
  return o;
}

bi::Expression* bi::Resolver::modify(LambdaFunction* o) {
  ++inLambda;
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  --inLambda;
  return o;
}

bi::Expression* bi::Resolver::modify(NamedExpression* o) {
  Modifier::modify(o);
  lookup(o);
  return o;
}

bi::Statement* bi::Resolver::modify(GlobalVariable* o) {
  if (stage == RESOLVER_HEADER) {
    o->type = o->type->accept(this);
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    o->brackets = o->brackets->accept(this);
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(MemberVariable* o) {
  if (stage == RESOLVER_HEADER) {
    o->type = o->type->accept(this);
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    o->brackets = o->brackets->accept(this);
    o->args = o->args->accept(this);
    o->value = o->value->accept(this);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(LocalVariable* o) {
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
    scopes.pop_back();
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
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
    scopes.pop_back();
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  --inFiber;
  return o;
}

bi::Statement* bi::Resolver::modify(Program* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->params = o->params->accept(this);
    scopes.pop_back();
    scopes.back()->add(o);
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
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
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
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  --inFiber;
  return o;
}

bi::Statement* bi::Resolver::modify(BinaryOperator* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->left = o->left->accept(this);
    o->right = o->right->accept(this);
    o->returnType = o->returnType->accept(this);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(UnaryOperator* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->single = o->single->accept(this);
    o->returnType = o->returnType->accept(this);
    scopes.pop_back();
    scopes.back()->add(o);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
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
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(ConversionOperator* o) {
  if (stage == RESOLVER_HEADER) {
    o->returnType = o->returnType->accept(this);
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Class* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.push_back(o->scope);
    o->typeParams = o->typeParams->accept(this);
    o->base = o->base->accept(this);
    scopes.push_back(o->initScope);
    o->params = o->params->accept(this);
    scopes.pop_back();
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  } else if (stage == RESOLVER_SOURCE) {
    scopes.push_back(o->scope);
    scopes.push_back(o->initScope);
    o->args = o->args->accept(this);
    scopes.pop_back();
    o->braces = o->braces->accept(this);
    scopes.pop_back();
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Basic* o) {
  if (stage == RESOLVER_HEADER) {
    scopes.back()->add(o);
    o->base = o->base->accept(this);
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
  return o;
}

bi::Statement* bi::Resolver::modify(For* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::Resolver::modify(Parallel* o) {
  ++inParallel;
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  --inParallel;
  return o;
}

bi::Statement* bi::Resolver::modify(While* o) {
  scopes.push_back(o->scope);
  Modifier::modify(o);
  scopes.pop_back();
  return o;
}

bi::Statement* bi::Resolver::modify(DoWhile* o) {
  scopes.push_back(o->scope);
  o->braces = o->braces->accept(this);
  scopes.pop_back();
  o->cond = o->cond->accept(this);
  return o;
}

bi::Type* bi::Resolver::modify(NamedType* o) {
  Modifier::modify(o);
  lookup(o);
  return o;
}

void bi::Resolver::lookup(NamedExpression* o) {
  ///@todo
//  bool found = false;
//  for (auto iter = scopes.rbegin(); !o->category && iter != scopes.rend(); ++iter) {
//    auto scope = *iter;
//    if (scope->lookup(o)) {
//      o->category = scope->category;
//    }
//  }
}

void bi::Resolver::lookup(NamedType* o) {
  ///@todo
//  bool found = false;
//  for (auto iter = scopes.rbegin(); !o->category && iter != scopes.rend(); ++iter) {
//    auto scope = *iter;
//    if (scope->lookup(o)) {
//      o->category = scope->category;
//    }
//  }
}
