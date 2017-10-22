/**
 * @file
 */
#pragma once

#include "bi/visitor/Resolver.hpp"

#include <stack>

namespace bi {
/**
 * This is the fourth pass of the abstract syntax tree after parsing, handling
 * definitions.
 *
 * @ingroup compiler_visitor
 */
class ResolverSource: public Resolver {
public:
  /**
   * Constructor.
   */
  ResolverSource();

  /**
   * Destructor.
   */
  virtual ~ResolverSource();

  using Resolver::modify;

  virtual Expression* modify(List<Expression>* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Brackets* o);
  virtual Expression* modify(Cast* o);
  virtual Expression* modify(Call* o);
  virtual Expression* modify(BinaryCall* o);
  virtual Expression* modify(UnaryCall* o);
  virtual Expression* modify(Slice* o);
  virtual Expression* modify(Query* o);
  virtual Expression* modify(Get* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Nil* o);
  virtual Expression* modify(LocalVariable* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(MemberParameter* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<MemberParameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(OverloadedIdentifier<Function>* o);
  virtual Expression* modify(OverloadedIdentifier<Fiber>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFunction>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFiber>* o);
  virtual Expression* modify(OverloadedIdentifier<BinaryOperator>* o);
  virtual Expression* modify(OverloadedIdentifier<UnaryOperator>* o);

  virtual Statement* modify(Assignment* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);

private:
  /**
   * Generic implementation of modify() for variable identifiers.
   */
  template<class ObjectType>
  Identifier<ObjectType>* modifyVariableIdentifier(
      bi::Identifier<ObjectType>* o);

  /**
   * Generic implementation of modify() for function identifiers.
   */
  template<class ObjectType>
  OverloadedIdentifier<ObjectType>* modifyFunctionIdentifier(
      bi::OverloadedIdentifier<ObjectType>* o);

  /**
   * Return type of current function. Stack as functions may contain lambda
   * functions may contain lambda functions...
   */
  std::stack<Type*> returnTypes;

  /**
   * Yield type of current fiber.
   */
  Type* currentYieldType;
};
}

template<class ObjectType>
bi::Identifier<ObjectType>* bi::ResolverSource::modifyVariableIdentifier(
    bi::Identifier<ObjectType>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>* bi::ResolverSource::modifyFunctionIdentifier(
    bi::OverloadedIdentifier<ObjectType>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = new OverloadedType(o->target, o->loc);
  return o;
}
