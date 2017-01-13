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

  virtual bool dispatch(Expression& o);
  virtual bool le(BracesExpression& o);
  virtual bool le(BracketsExpression& o);
  virtual bool le(EmptyExpression& o);
  virtual bool le(List<Expression>& o);
  virtual bool le(FuncParameter& o);
  virtual bool le(FuncReference& o);
  virtual bool le(Literal<unsigned char>& o);
  virtual bool le(Literal<int64_t>& o);
  virtual bool le(Literal<double>& o);
  virtual bool le(Literal<const char*>& o);
  virtual bool le(Member& o);
  virtual bool le(ParenthesesExpression& o);
  virtual bool le(RandomInit& o);
  virtual bool le(Range& o);
  virtual bool le(This& o);
  virtual bool le(VarParameter& o);
  virtual bool le(VarReference& o);
};
}
