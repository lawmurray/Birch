/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
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
  ParenthesesExpression(Expression* single, shared_ptr<Location> loc = nullptr);

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

  virtual possibly dispatch(Expression& o);
  virtual possibly le(BracesExpression& o);
  virtual possibly le(BracketsExpression& o);
  virtual possibly le(EmptyExpression& o);
  virtual possibly le(List<Expression>& o);
  virtual possibly le(FuncParameter& o);
  virtual possibly le(FuncReference& o);
  virtual possibly le(Literal<unsigned char>& o);
  virtual possibly le(Literal<int64_t>& o);
  virtual possibly le(Literal<double>& o);
  virtual possibly le(Literal<const char*>& o);
  virtual possibly le(Member& o);
  virtual possibly le(ParenthesesExpression& o);
  virtual possibly le(RandomInit& o);
  virtual possibly le(Range& o);
  virtual possibly le(This& o);
  virtual possibly le(VarParameter& o);
  virtual possibly le(VarReference& o);
};
}
