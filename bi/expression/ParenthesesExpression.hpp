/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
class Parameter;
class VarReference;

/**
 * ParenthesesExpression.
 *
 * @ingroup compiler_expression
 */
class ParenthesesExpression: public Expression, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in parentheses.
   * @param loc Location.
   */
  ParenthesesExpression(Expression* single = new EmptyExpression(),
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ParenthesesExpression();

  /**
   * Strip parentheses.
   */
  virtual Expression* strip();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const BracketsExpression& o) const;
  virtual bool definitely(const EmptyExpression& o) const;
  virtual bool definitely(const Identifier<Parameter>& o) const;
  virtual bool definitely(const Identifier<GlobalVariable>& o) const;
  virtual bool definitely(const Identifier<LocalVariable>& o) const;
  virtual bool definitely(const Identifier<MemberVariable>& o) const;
  virtual bool definitely(const Identifier<Function>& o) const;
  virtual bool definitely(const Identifier<Coroutine>& o) const;
  virtual bool definitely(const Identifier<MemberFunction>& o) const;
  virtual bool definitely(const Identifier<BinaryOperator>& o) const;
  virtual bool definitely(const Identifier<UnaryOperator>& o) const;
  virtual bool definitely(const Index& o) const;
  virtual bool definitely(const LambdaFunction& o) const;
  virtual bool definitely(const List<Expression>& o) const;
  virtual bool definitely(const Literal<bool>& o);
  virtual bool definitely(const Literal<int64_t>& o);
  virtual bool definitely(const Literal<double>& o) const;
  virtual bool definitely(const Literal<const char*>& o);
  virtual bool definitely(const Member& o) const;
  virtual bool definitely(const Parameter& o) const;
  virtual bool definitely(const ParenthesesExpression& o) const;
  virtual bool definitely(const Range& o) const;
  virtual bool definitely(const Span& o) const;
  virtual bool definitely(const Super& o) const;
  virtual bool definitely(const This& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const BracketsExpression& o) const;
  virtual bool possibly(const EmptyExpression& o) const;
  virtual bool possibly(const Identifier<Parameter>& o) const;
  virtual bool possibly(const Identifier<GlobalVariable>& o) const;
  virtual bool possibly(const Identifier<LocalVariable>& o) const;
  virtual bool possibly(const Identifier<MemberVariable>& o) const;
  virtual bool possibly(const Identifier<Function>& o) const;
  virtual bool possibly(const Identifier<Coroutine>& o) const;
  virtual bool possibly(const Identifier<MemberFunction>& o) const;
  virtual bool possibly(const Identifier<BinaryOperator>& o) const;
  virtual bool possibly(const Identifier<UnaryOperator>& o) const;
  virtual bool possibly(const Index& o) const;
  virtual bool possibly(const LambdaFunction& o) const;
  virtual bool possibly(const List<Expression>& o) const;
  virtual bool possibly(const Literal<bool>& o);
  virtual bool possibly(const Literal<int64_t>& o);
  virtual bool possibly(const Literal<double>& o) const;
  virtual bool possibly(const Literal<const char*>& o);
  virtual bool possibly(const Member& o) const;
  virtual bool possibly(const Parameter& o) const;
  virtual bool possibly(const ParenthesesExpression& o) const;
  virtual bool possibly(const Range& o) const;
  virtual bool possibly(const Span& o) const;
  virtual bool possibly(const Super& o) const;
  virtual bool possibly(const This& o) const;
};
}
