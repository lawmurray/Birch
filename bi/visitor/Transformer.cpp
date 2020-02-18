/**
 * @file
 */
#include "bi/visitor/Transformer.hpp"

#include "bi/visitor/Cloner.hpp"

bi::Statement* bi::Transformer::modify(Assume* o) {
  Cloner cloner;

  Statement* result = nullptr;
  if (*o->name == "<-?") {
    auto tmp = new LocalVariable(o->right, o->loc);
    auto ref = new NamedExpression(tmp->name, o->loc);
    auto cond = new Query(ref->accept(&cloner), o->loc);
    auto trueBranch = new ExpressionStatement(
        new Assign(o->left, new Name("<-"),
            new Get(ref->accept(&cloner), o->loc), o->loc), o->loc);
    auto falseBranch = new EmptyStatement(o->loc);
    auto conditional = new If(cond, trueBranch, falseBranch, o->loc);
    result = new StatementList(tmp, conditional, o->loc);
  } else {
    o->right = o->right->accept(this);
    if (*o->name == "<~") {
      auto identifier = new NamedExpression(new Name("SimulateEvent"),
          new EmptyType(o->loc), o->loc);
      auto call = new Call(identifier, o->right->accept(&cloner));
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
      auto args = new ExpressionList(o->left, o->right->accept(&cloner),
          o->loc);
      result = new Yield(new Call(identifier, args, o->loc), o->loc);
    } else if (*o->name == "~") {
      auto identifier = new NamedExpression(new Name("AssumeEvent"),
          new EmptyType(o->loc), o->loc);
      auto args = new ExpressionList(o->left, o->right->accept(&cloner),
          o->loc);
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
  return Modifier::modify(o);
}

bi::Statement* bi::Transformer::modify(ExpressionStatement* o) {
  /* when in the body of a fiber and another fiber is called while ignoring
   * its return type, this is syntactic sugar for a loop */
//  auto fiberCall = dynamic_cast<Call*>(o->single);
//  auto memberFiberCall = dynamic_cast<Call*>(o->single);
//  if (fiberCall || memberFiberCall) {
//    auto name = new Name();
//    auto var = new LocalVariable(AUTO, name, new EmptyType(o->loc),
//        new EmptyExpression(o->loc), new EmptyExpression(o->loc), o->single,
//        o->loc);
//    auto query = new Query(new NamedExpression(name, o->loc), o->loc);
//    auto get = new Get(new NamedExpression(name, o->loc), o->loc);
//    auto yield = new Yield(get, o->loc);
//    auto loop = new While(new Parentheses(query, o->loc),
//        new Braces(yield, o->loc), o->loc);
//    auto result = new StatementList(var, loop, o->loc);
//
//    return result->accept(this);
//  } else {
    return Modifier::modify(o);
//  }
}
