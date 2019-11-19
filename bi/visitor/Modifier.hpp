/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Modifying visitor.
 *
 * @ingroup visitor
 */
class Modifier {
public:
  /**
   * Destructor.
   */
  virtual ~Modifier();

  virtual Package* modify(Package* o);
  virtual File* modify(File* o);

  virtual Expression* modify(EmptyExpression* o);
  virtual Expression* modify(ExpressionList* o);
  virtual Expression* modify(Literal<bool>* o);
  virtual Expression* modify(Literal<int64_t>* o);
  virtual Expression* modify(Literal<double>* o);
  virtual Expression* modify(Literal<const char*>* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Sequence* o);
  virtual Expression* modify(Binary* o);
  virtual Expression* modify(Cast* o);
  virtual Expression* modify(Call<Unknown>* o);
  virtual Expression* modify(Call<Function>* o);
  virtual Expression* modify(Call<MemberFunction>* o);
  virtual Expression* modify(Call<Fiber>* o);
  virtual Expression* modify(Call<MemberFiber>* o);
  virtual Expression* modify(Call<Parameter>* o);
  virtual Expression* modify(Call<LocalVariable>* o);
  virtual Expression* modify(Call<MemberVariable>* o);
  virtual Expression* modify(Call<GlobalVariable>* o);
  virtual Expression* modify(Call<BinaryOperator>* o);
  virtual Expression* modify(Call<UnaryOperator>* o);
  virtual Expression* modify(Assign* o);
  virtual Expression* modify(Slice* o);
  virtual Expression* modify(Query* o);
  virtual Expression* modify(Get* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Global* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Nil* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(Generic* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<ForVariable>* o);
  virtual Expression* modify(OverloadedIdentifier<Unknown>* o);
  virtual Expression* modify(OverloadedIdentifier<Function>* o);
  virtual Expression* modify(OverloadedIdentifier<Fiber>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFunction>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFiber>* o);
  virtual Expression* modify(OverloadedIdentifier<BinaryOperator>* o);
  virtual Expression* modify(OverloadedIdentifier<UnaryOperator>* o);

  virtual Statement* modify(EmptyStatement* o);
  virtual Statement* modify(Braces* o);
  virtual Statement* modify(StatementList* o);
  virtual Statement* modify(Assume* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(ForVariable* o);
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
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(DoWhile* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);
  virtual Statement* modify(Raw* o);
  virtual Statement* modify(Instantiated<Type>* o);
  virtual Statement* modify(Instantiated<Expression>* o);

  virtual Type* modify(EmptyType* o);
  virtual Type* modify(TypeList* o);
  virtual Type* modify(UnknownType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(GenericType* o);
  virtual Type* modify(MemberType* o);
  virtual Type* modify(ArrayType* o);
  virtual Type* modify(TupleType* o);
  virtual Type* modify(BinaryType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(FiberType* o);
  virtual Type* modify(OptionalType* o);
  virtual Type* modify(NilType* o);
};
}
