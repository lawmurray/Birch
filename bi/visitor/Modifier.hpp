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
 * @ingroup compiler_visitor
 */
class Modifier {
public:
  /**
   * Destructor.
   */
  virtual ~Modifier();

  virtual void modify(File* o);
  virtual Expression* modify(EmptyExpression* o);
  virtual Expression* modify(List<Expression>* o);
  virtual Expression* modify(Literal<bool>* o);
  virtual Expression* modify(Literal<int64_t>* o);
  virtual Expression* modify(Literal<double>* o);
  virtual Expression* modify(Literal<const char*>* o);
  virtual Expression* modify(ParenthesesExpression* o);
  virtual Expression* modify(BracketsExpression* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(MemberParameter* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<MemberParameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(Identifier<Function>* o);
  virtual Expression* modify(Identifier<Coroutine>* o);
  virtual Expression* modify(Identifier<MemberFunction>* o);
  virtual Expression* modify(Identifier<BinaryOperator>* o);
  virtual Expression* modify(Identifier<UnaryOperator>* o);

  virtual Statement* modify(EmptyStatement* o);
  virtual Statement* modify(List<Statement>* o);
  virtual Statement* modify(Assignment* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Coroutine* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Alias* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Import* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Raw* o);

  virtual Type* modify(EmptyType* o);
  virtual Type* modify(List<Type>* o);
  virtual Type* modify(IdentifierType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(ArrayType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(FiberType* o);
};
}
