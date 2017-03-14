/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
class VarParameter;
class VarReference;

/**
 * ParenthesesExpression.
 *
 * @ingroup compiler_expression
 */
class ParenthesesExpression: public Expression, public ExpressionUnary {
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

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(BracesExpression& o);
  virtual bool definitely(BracketsExpression& o);
  virtual bool definitely(EmptyExpression& o);
  virtual bool definitely(List<Expression>& o);
  virtual bool definitely(FuncParameter& o);
  virtual bool definitely(FuncReference& o);
  virtual bool definitely(Literal<unsigned char>& o);
  virtual bool definitely(Literal<int64_t>& o);
  virtual bool definitely(Literal<double>& o);
  virtual bool definitely(Literal<const char*>& o);
  virtual bool definitely(Member& o);
  virtual bool definitely(ParenthesesExpression& o);
  virtual bool definitely(RandomInit& o);
  virtual bool definitely(Range& o);
  virtual bool definitely(This& o);
  virtual bool definitely(VarParameter& o);
  virtual bool definitely(VarReference& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(BracesExpression& o);
  virtual bool possibly(BracketsExpression& o);
  virtual bool possibly(EmptyExpression& o);
  virtual bool possibly(List<Expression>& o);
  virtual bool possibly(FuncParameter& o);
  virtual bool possibly(FuncReference& o);
  virtual bool possibly(Literal<unsigned char>& o);
  virtual bool possibly(Literal<int64_t>& o);
  virtual bool possibly(Literal<double>& o);
  virtual bool possibly(Literal<const char*>& o);
  virtual bool possibly(Member& o);
  virtual bool possibly(ParenthesesExpression& o);
  virtual bool possibly(RandomInit& o);
  virtual bool possibly(Range& o);
  virtual bool possibly(This& o);
  virtual bool possibly(VarParameter& o);
  virtual bool possibly(VarReference& o);
};
}
