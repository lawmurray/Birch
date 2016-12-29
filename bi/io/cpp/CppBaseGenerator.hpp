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

  virtual void visit(const BoolLiteral* o);
  virtual void visit(const IntLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const Name* o);

  virtual void visit(const ExpressionList* o);
  virtual void visit(const StatementList* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Range* o);
  virtual void visit(const Traversal* o);
  virtual void visit(const This* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const ModelReference* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);

  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const Conditional* o);
  virtual void visit(const Loop* o);
  virtual void visit(const Raw* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const ParenthesesType* o);

protected:
  /**
   * Output header instead of source?
   */
  bool header;

};
}
