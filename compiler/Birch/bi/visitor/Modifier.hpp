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
  virtual Expression* modify(Cast* o);
  virtual Expression* modify(Call* o);
  virtual Expression* modify(BinaryCall* o);
  virtual Expression* modify(UnaryCall* o);
  virtual Expression* modify(Assign* o);
  virtual Expression* modify(Slice* o);
  virtual Expression* modify(Query* o);
  virtual Expression* modify(Get* o);
  virtual Expression* modify(GetReturn* o);
  virtual Expression* modify(Spin* o);
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
  virtual Expression* modify(NamedExpression* o);

  virtual Statement* modify(EmptyStatement* o);
  virtual Statement* modify(Braces* o);
  virtual Statement* modify(StatementList* o);
  virtual Statement* modify(Assume* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(LocalVariable* o);
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
  virtual Statement* modify(Parallel* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(DoWhile* o);
  virtual Statement* modify(Block* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);
  virtual Statement* modify(Raw* o);

  virtual Type* modify(EmptyType* o);
  virtual Type* modify(TypeList* o);
  virtual Type* modify(NamedType* o);
  virtual Type* modify(MemberType* o);
  virtual Type* modify(ArrayType* o);
  virtual Type* modify(TupleType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(FiberType* o);
  virtual Type* modify(OptionalType* o);
};
}
