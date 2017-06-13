/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * Output stream for *.bi files.
 *
 * @ingroup compiler_io
 */
class bi_ostream: public indentable_ostream {
public:
  bi_ostream(std::ostream& base, const int level = 0, const bool header =
      false);

  using indentable_ostream::visit;

  virtual void visit(const File* o);
  virtual void visit(const Name* o);
  virtual void visit(const Path* o);

  virtual void visit(const List<Expression>* o);
  virtual void visit(const BooleanLiteral* o);
  virtual void visit(const IntegerLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Parameter* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const BinaryReference* o);
  virtual void visit(const UnaryReference* o);

  virtual void visit(const List<Statement>* o);
  virtual void visit(const Assignment* o);
  virtual void visit(const Function* o);
  virtual void visit(const Coroutine* o);
  virtual void visit(const Program* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const Import* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Return* o);
  virtual void visit(const Raw* o);

  virtual void visit(const List<Type>* o);
  virtual void visit(const TypeReference* o);
  virtual void visit(const TypeParameter* o);
  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const CoroutineType* o);
};
}
