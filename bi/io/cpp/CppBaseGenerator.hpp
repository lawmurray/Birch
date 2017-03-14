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

  virtual void visit(const BooleanLiteral* o);
  virtual void visit(const IntegerLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const Name* o);

  virtual void visit(const ExpressionList* o);
  virtual void visit(const StatementList* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const This* o);
  virtual void visit(const RandomInit* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const ModelReference* o);

  virtual void visit(const VarParameter* o);

  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const Conditional* o);
  virtual void visit(const Loop* o);
  virtual void visit(const Raw* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const RandomType* o);
  virtual void visit(const LambdaType* o);
  virtual void visit(const VariantType* o);

protected:
  /**
   * Generate the capture for a lambda.
   */
  void genCapture(const Expression* o);

  /*
   * Generate function calls of various kinds.
   */
  void genCallFunction(FuncReference* o);
  void genCallBinary(FuncReference* o);
  void genCallUnary(FuncReference* o);
  void genCallDispatcher(FuncReference* o);

  /**
   * Generate a single argument for a function call.
   *
   * @param arg Argument.
   * @param param Associated parameter.
   */
  void genArg(Expression* arg, VarParameter* param);

  /**
   * Output header instead of source?
   */
  bool header;

  /**
   * Currently generating type for an array?
   */
  bool inArray;
};
}
