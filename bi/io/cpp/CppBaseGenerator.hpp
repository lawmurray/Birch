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

  virtual void visit(const BooleanLiteral* o);
  virtual void visit(const IntegerLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);

  virtual void visit(const ExpressionList* o);
  virtual void visit(const StatementList* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Span* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const TypeReference* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);

  virtual void visit(const VarDeclaration* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Return* o);
  virtual void visit(const Raw* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const LambdaType* o);

protected:
  /**
   * Generate the capture for a lambda.
   */
  void genCapture(const Expression* o);

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

  /**
   * Are we in a coroutine body?
   */
  int inCoroutine;
};
}
