/**
 * @file
 */
#pragma once

#include "bi/common/Typed.hpp"
#include "bi/common/Located.hpp"

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
  bool operator<=(Expression& o);
  bool operator==(Expression& o);
  virtual bool dispatch(Expression& o) = 0;
  virtual bool le(BracesExpression& o);
  virtual bool le(BracketsExpression& o);
  virtual bool le(EmptyExpression& o);
  virtual bool le(List<Expression>& o);
  virtual bool le(FuncParameter& o);
  virtual bool le(FuncReference& o);
  virtual bool le(Literal<bool>& o);
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
