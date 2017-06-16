/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for within-function code.
 *
 * @ingroup compiler_io
 */
class CppBaseGenerator: public indentable_ostream {
public:
  CppBaseGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const Name* o);

  virtual void visit(const List<Expression>* o);
  virtual void visit(const Literal<bool>* o);
  virtual void visit(const Literal<int64_t>* o);
  virtual void visit(const Literal<double>* o);
  virtual void visit(const Literal<const char*>* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const LambdaFunction* o);
  virtual void visit(const Span* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);
  virtual void visit(const Identifier<Parameter>* o);
  virtual void visit(const Identifier<GlobalVariable>* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const Identifier<MemberVariable>* o);
  virtual void visit(const Identifier<Function>* o);
  virtual void visit(const Identifier<Coroutine>* o);
  virtual void visit(const Identifier<MemberFunction>* o);
  virtual void visit(const Identifier<BinaryOperator>* o);
  virtual void visit(const Identifier<UnaryOperator>* o);

  virtual void visit(const Parameter* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const List<Statement>* o);
  virtual void visit(const Assignment* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Return* o);
  virtual void visit(const Raw* o);

  virtual void visit(const List<Type>* o);
  virtual void visit(const TypeReference* o);
  virtual void visit(const EmptyType* o);
  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const CoroutineType* o);

protected:
  /**
   * Generate a built-in type.
   */
  void genBuiltin(const TypeReference* o);

  /**
   * Generate an argument to a function.
   *
   * @param arg The argument.
   * @param param The parameter.
   */
  void genArg(const Expression* arg, const Expression* param);

  /**
   * Output header instead of source?
   */
  bool header;

  /**
   * Are we in a return type?
   */
  int inReturn;
};
}
