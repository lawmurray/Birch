/**
 * @file
 */
#include "bi/visitor/Transformer.hpp"

#include "bi/visitor/all.hpp"

bi::Statement* bi::Transformer::modify(Assume* o) {
  Cloner cloner;

  Statement* result = nullptr;
  if (*o->name == "<-?") {
    auto tmp = new LocalVariable(o->right, o->loc);
    auto ref = new NamedExpression(tmp->name, o->loc);
    auto cond = new Query(ref->accept(&cloner), o->loc);
    auto trueBranch = new Braces(new ExpressionStatement(
        new Assign(o->left, new Name("<-"),
            new Get(ref->accept(&cloner), o->loc), o->loc), o->loc), o->loc);
    auto falseBranch = new EmptyStatement(o->loc);
    auto conditional = new If(cond, trueBranch, falseBranch, o->loc);
    result = new StatementList(tmp, conditional, o->loc);
  } else {
    o->right = o->right->accept(this);
    if (*o->name == "<~") {
      auto identifier = new NamedExpression(new Name("SimulateEvent"),
          new EmptyType(o->loc), o->loc);
      auto distribution = new Call(new Member(o->right->accept(&cloner),
          new NamedExpression(new Name("distribution"), new EmptyType(),
          o->loc), o->loc), o->loc);
      auto call = new Call(identifier, distribution);
      auto tmp = new LocalVariable(call, o->loc);
      auto yield = new Yield(new NamedExpression(tmp->name, o->loc), o->loc);
      auto member = new Member(new NamedExpression(tmp->name, o->loc),
          new NamedExpression(new Name("value"), new EmptyType(), o->loc),
          o->loc);
      auto value = new Call(member, new EmptyExpression(), o->loc);
      auto assign = new ExpressionStatement(new Assign(o->left,
          new Name("<-"), value, o->loc), o->loc);
      result = new StatementList(tmp, new StatementList(yield, assign,
          o->loc), o->loc);
    } else if (*o->name == "~>") {
      auto identifier = new NamedExpression(new Name("ObserveEvent"),
          new EmptyType(o->loc), o->loc);
      auto distribution = new Call(new Member(o->right->accept(&cloner),
          new NamedExpression(new Name("distribution"), new EmptyType(),
          o->loc), o->loc), o->loc);
      auto args = new ExpressionList(o->left, distribution, o->loc);
      result = new Yield(new Call(identifier, args, o->loc), o->loc);
    } else if (*o->name == "~") {
      auto identifier = new NamedExpression(new Name("AssumeEvent"),
          new EmptyType(o->loc), o->loc);
      auto distribution = new Call(new Member(o->right->accept(&cloner),
          new NamedExpression(new Name("distribution"), new EmptyType(),
          o->loc), o->loc), o->loc);
      auto args = new ExpressionList(o->left, distribution, o->loc);
      result = new Yield(new Call(identifier, args, o->loc), o->loc);
    } else {
      assert(false);
    }
  }
  return result->accept(this);
}

bi::Statement* bi::Transformer::modify(Class* o) {
  if (o->base->isEmpty() && o->name->str() != "Object") {
    /* if the class derives from nothing else, then derive from Object,
     * unless this is itself the declaration of the Object class */
    o->base = new NamedType(false, new Name("Object"), new EmptyType(), o->loc);
  }
  return ContextualModifier::modify(o);
}

bi::Statement* bi::Transformer::modify(Fiber* o) {
  ContextualModifier::modify(o);
  createStart(o);
  createResumes(o);
  return o;
}

bi::Statement* bi::Transformer::modify(MemberFiber* o) {
  ContextualModifier::modify(o);
  createStart(o);
  createResumes(o);
  return o;
}

void bi::Transformer::insertReturn(Statement* o) {
  auto function = dynamic_cast<Function*>(o);
  auto fiberType = dynamic_cast<FiberType*>(function->returnType);
  assert(fiberType);
  if (fiberType->returnType->isEmpty()) {
    /* iterate to the last statement */
    auto iter = function->braces->begin();
    for (auto i = 0; i < function->braces->count() - 1; ++i) {
      ++iter;
    }

    /* if that statement is not a return, add one */
    if (!dynamic_cast<const Return*>(*iter)) {
      function->braces = new StatementList(function->braces, new Return(
          new EmptyExpression(function->loc), function->loc), function->loc);
    }
  }
}

void bi::Transformer::createStart(Fiber* o) {
  Resumer resumer;
  o->start = o->accept(&resumer)->accept(this);
  insertReturn(o->start);
}

void bi::Transformer::createResumes(Fiber* o) {
  Gatherer<Yield> yields;
  o->accept(&yields);
  for (auto yield : yields) {
    Resumer resumer(yield);
    yield->resume = o->accept(&resumer)->accept(this);
    insertReturn(yield->resume);
  }
}
