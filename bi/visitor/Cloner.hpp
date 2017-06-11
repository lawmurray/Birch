/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Cloning visitor.
 *
 * @ingroup compiler_visitor
 */
class Cloner {
public:
  /**
   * Destructor.
   */
  virtual ~Cloner();

  virtual Expression* clone(const EmptyExpression* o);
  virtual Statement* clone(const EmptyStatement* o);
  virtual Type* clone(const EmptyType* o);

  virtual Expression* clone(const BooleanLiteral* o);
  virtual Expression* clone(const IntegerLiteral* o);
  virtual Expression* clone(const RealLiteral* o);
  virtual Expression* clone(const StringLiteral* o);
  virtual Expression* clone(const List<Expression>* o);
  virtual Statement* clone(const List<Statement>* o);
  virtual Expression* clone(const ParenthesesExpression* o);
  virtual Expression* clone(const BracesExpression* o);
  virtual Expression* clone(const BracketsExpression* o);
  virtual Expression* clone(const Span* o);
  virtual Expression* clone(const Index* o);
  virtual Expression* clone(const Range* o);
  virtual Expression* clone(const Member* o);
  virtual Expression* clone(const Super* o);
  virtual Expression* clone(const This* o);

  virtual Expression* clone(const VarReference* o);
  virtual Expression* clone(const FuncReference* o);
  virtual Expression* clone(const BinaryReference* o);
  virtual Type* clone(const TypeReference* o);
  virtual Expression* clone(const ProgReference* o);

  virtual Expression* clone(const VarParameter* o);
  virtual Expression* clone(const FuncParameter* o);
  virtual Expression* clone(const BinaryParameter* o);
  virtual Expression* clone(const ConversionParameter* o);
  virtual Type* clone(const TypeParameter* o);
  virtual Expression* clone(const ProgParameter* o);

  virtual File* clone(const File* o);
  virtual Statement* clone(const Import* o);
  virtual Statement* clone(const ExpressionStatement* o);
  virtual Statement* clone(const If* o);
  virtual Statement* clone(const For* o);
  virtual Statement* clone(const While* o);
  virtual Statement* clone(const Return* o);
  virtual Statement* clone(const Raw* o);
  virtual Statement* clone(const Declaration<Expression>* o);
  virtual Statement* clone(const Declaration<Type>* o);

  virtual Type* clone(const BracketsType* o);
  virtual Type* clone(const ParenthesesType* o);
  virtual Type* clone(const FunctionType* o);
  virtual Type* clone(const CoroutineType* o);
  virtual Type* clone(const List<Type>* o);
};
}
