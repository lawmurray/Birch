/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Visitor.
 *
 * @ingroup compiler_visitor
 */
class Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~Visitor();

  virtual void visit(const File* o);
  virtual void visit(const Name* o);
  virtual void visit(const Path* o);

  virtual void visit(const EmptyExpression* o);
  virtual void visit(const List<Expression>* o);
  virtual void visit(const Literal<bool>* o);
  virtual void visit(const Literal<int64_t>* o);
  virtual void visit(const Literal<double>* o);
  virtual void visit(const Literal<const char*>* o);
  virtual void visit(const Parentheses* o);
  virtual void visit(const Brackets* o);
  virtual void visit(const Binary* o);
  virtual void visit(const Call* o);
  virtual void visit(const BinaryCall* o);
  virtual void visit(const UnaryCall* o);
  virtual void visit(const Slice* o);
  virtual void visit(const Query* o);
  virtual void visit(const Get* o);
  virtual void visit(const LambdaFunction* o);
  virtual void visit(const Index* o);
  virtual void visit(const Span* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);
  virtual void visit(const Nil* o);
  virtual void visit(const Parameter* o);
  virtual void visit(const MemberParameter* o);
  virtual void visit(const Identifier<Unknown>* o);
  virtual void visit(const Identifier<Parameter>* o);
  virtual void visit(const Identifier<MemberParameter>* o);
  virtual void visit(const Identifier<GlobalVariable>* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const Identifier<MemberVariable>* o);
  virtual void visit(const OverloadedIdentifier<Function>* o);
  virtual void visit(const OverloadedIdentifier<Fiber>* o);
  virtual void visit(const OverloadedIdentifier<MemberFunction>* o);
  virtual void visit(const OverloadedIdentifier<MemberFiber>* o);
  virtual void visit(const OverloadedIdentifier<BinaryOperator>* o);
  virtual void visit(const OverloadedIdentifier<UnaryOperator>* o);

  virtual void visit(const EmptyStatement* o);
  virtual void visit(const List<Statement>* o);
  virtual void visit(const Assignment* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const Fiber* o);
  virtual void visit(const Program* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const MemberFiber* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const Class* o);
  virtual void visit(const Alias* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Import* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Assert* o);
  virtual void visit(const Return* o);
  virtual void visit(const Yield* o);
  virtual void visit(const Raw* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const ListType* o);
  virtual void visit(const IdentifierType* o);
  virtual void visit(const ClassType* o);
  virtual void visit(const AliasType* o);
  virtual void visit(const BasicType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const BinaryType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const OverloadedType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const OptionalType* o);
  virtual void visit(const NilType* o);
};
}
