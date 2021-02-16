/**
 * @file
 */
#pragma once

#include "src/expression/all.hpp"
#include "src/statement/all.hpp"
#include "src/type/all.hpp"

namespace birch {
/**
 * Cloning visitor.
 *
 * @ingroup visitor
 */
class Cloner {
public:
  /**
   * Destructor.
   */
  virtual ~Cloner();

  virtual Package* clone(const Package* o);
  virtual File* clone(const File* o);

  virtual Expression* clone(const EmptyExpression* o);
  virtual Expression* clone(const ExpressionList* o);
  virtual Expression* clone(const Literal<bool>* o);
  virtual Expression* clone(const Literal<int64_t>* o);
  virtual Expression* clone(const Literal<double>* o);
  virtual Expression* clone(const Literal<const char*>* o);
  virtual Expression* clone(const Parentheses* o);
  virtual Expression* clone(const Sequence* o);
  virtual Expression* clone(const Cast* o);
  virtual Expression* clone(const Call* o);
  virtual Expression* clone(const BinaryCall* o);
  virtual Expression* clone(const UnaryCall* o);
  virtual Expression* clone(const Assign* o);
  virtual Expression* clone(const Slice* o);
  virtual Expression* clone(const Query* o);
  virtual Expression* clone(const Get* o);
  virtual Expression* clone(const LambdaFunction* o);
  virtual Expression* clone(const Span* o);
  virtual Expression* clone(const Range* o);
  virtual Expression* clone(const Member* o);
  virtual Expression* clone(const Global* o);
  virtual Expression* clone(const Super* o);
  virtual Expression* clone(const This* o);
  virtual Expression* clone(const Nil* o);
  virtual Expression* clone(const Parameter* o);
  virtual Expression* clone(const Generic* o);
  virtual Expression* clone(const NamedExpression* o);

  virtual Statement* clone(const EmptyStatement* o);
  virtual Statement* clone(const Braces* o);
  virtual Statement* clone(const StatementList* o);
  virtual Statement* clone(const GlobalVariable* o);
  virtual Statement* clone(const MemberVariable* o);
  virtual Statement* clone(const LocalVariable* o);
  virtual Statement* clone(const Function* o);
  virtual Statement* clone(const Program* o);
  virtual Statement* clone(const MemberFunction* o);
  virtual Statement* clone(const BinaryOperator* o);
  virtual Statement* clone(const UnaryOperator* o);
  virtual Statement* clone(const AssignmentOperator* o);
  virtual Statement* clone(const ConversionOperator* o);
  virtual Statement* clone(const SliceOperator* o);
  virtual Statement* clone(const Class* o);
  virtual Statement* clone(const Basic* o);
  virtual Statement* clone(const ExpressionStatement* o);
  virtual Statement* clone(const If* o);
  virtual Statement* clone(const For* o);
  virtual Statement* clone(const Parallel* o);
  virtual Statement* clone(const While* o);
  virtual Statement* clone(const DoWhile* o);
  virtual Statement* clone(const With* o);
  virtual Statement* clone(const Block* o);
  virtual Statement* clone(const Assert* o);
  virtual Statement* clone(const Return* o);
  virtual Statement* clone(const Factor* o);
  virtual Statement* clone(const Raw* o);

  virtual Type* clone(const EmptyType* o);
  virtual Type* clone(const TypeList* o);
  virtual Type* clone(const NamedType* o);
  virtual Type* clone(const MemberType* o);
  virtual Type* clone(const ArrayType* o);
  virtual Type* clone(const TupleType* o);
  virtual Type* clone(const FunctionType* o);
  virtual Type* clone(const OptionalType* o);
};
}
