/**
 * @file
 */
#pragma once

#include "bi/common/Typed.hpp"
#include "bi/common/Located.hpp"
#include "bi/primitive/possibly.hpp"

#include <cassert>

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class BracesExpression;
class BracketsExpression;
class EmptyExpression;
template<class T> class List;
class FuncParameter;
class FuncReference;
template<class T> class Literal;
class Member;
class ParenthesesExpression;
class RandomInit;
class Range;
class This;
class VarParameter;
class VarReference;

/**
 * Expression.
 *
 * @ingroup compiler_expression
 */
class Expression: public Located, public Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   * @param loc Location.
   */
  Expression(Type* type, shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Expression(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) expression.
   */
  virtual Expression* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified expression.
   */
  virtual Expression* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is expression empty?
   */
  virtual bool isEmpty() const;

  /**
   * Is this a primary expression?
   */
  virtual bool isPrimary() const;

  /**
   * Is this a rich expression?
   */
  virtual bool isRich() const;

  /**
   * Strip parentheses.
   */
  virtual Expression* strip();

  /**
   * Number of expressions in tuple.
   */
  virtual int tupleSize() const;

  /**
   * Number of range expressions in tuple.
   */
  virtual int tupleDims() const;

  /*
   * Partial order comparison operator for comparing expressions in terms of
   * specialisation. These double-dispatch to the #le functions below, which
   * can be implemented for specific types in derived classes.
   */
  possibly operator<=(Expression& o);
  possibly operator==(Expression& o);
  virtual possibly dispatch(Expression& o) = 0;
  virtual possibly le(BracesExpression& o);
  virtual possibly le(BracketsExpression& o);
  virtual possibly le(EmptyExpression& o);
  virtual possibly le(List<Expression>& o);
  virtual possibly le(FuncParameter& o);
  virtual possibly le(FuncReference& o);
  virtual possibly le(Literal<bool>& o);
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
