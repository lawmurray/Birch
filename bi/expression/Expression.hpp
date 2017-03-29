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
template<class T> class Iterator;
template<class T> class List;
class FuncParameter;
class FuncReference;
class Index;
template<class T> class Literal;
class Member;
class ParenthesesExpression;
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
   * Iterator to first element if this is an expression list, otherwise to
   * itself.
   */
  Iterator<Expression> begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  Iterator<Expression> end() const;

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
   * Does this function have an assignable parameter?
   */
  virtual bool hasAssignable() const;

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
   * Double-dispatch partial order comparisons.
   */
  bool definitely(Expression& o);
  virtual bool dispatchDefinitely(Expression& o) = 0;
  virtual bool definitely(BracesExpression& o);
  virtual bool definitely(BracketsExpression& o);
  virtual bool definitely(EmptyExpression& o);
  virtual bool definitely(List<Expression>& o);
  virtual bool definitely(FuncParameter& o);
  virtual bool definitely(FuncReference& o);
  virtual bool definitely(Index& o);
  virtual bool definitely(Literal<bool>& o);
  virtual bool definitely(Literal<int64_t>& o);
  virtual bool definitely(Literal<double>& o);
  virtual bool definitely(Literal<const char*>& o);
  virtual bool definitely(Member& o);
  virtual bool definitely(ParenthesesExpression& o);
  virtual bool definitely(Range& o);
  virtual bool definitely(This& o);
  virtual bool definitely(VarParameter& o);
  virtual bool definitely(VarReference& o);

  bool possibly(Expression& o);
  virtual bool dispatchPossibly(Expression& o) = 0;
  virtual bool possibly(BracesExpression& o);
  virtual bool possibly(BracketsExpression& o);
  virtual bool possibly(EmptyExpression& o);
  virtual bool possibly(List<Expression>& o);
  virtual bool possibly(FuncParameter& o);
  virtual bool possibly(FuncReference& o);
  virtual bool possibly(Index& o);
  virtual bool possibly(Literal<bool>& o);
  virtual bool possibly(Literal<int64_t>& o);
  virtual bool possibly(Literal<double>& o);
  virtual bool possibly(Literal<const char*>& o);
  virtual bool possibly(Member& o);
  virtual bool possibly(ParenthesesExpression& o);
  virtual bool possibly(Range& o);
  virtual bool possibly(This& o);
  virtual bool possibly(VarParameter& o);
  virtual bool possibly(VarReference& o);

  /**
   * Operators for equality comparisons.
   */
  bool operator==(Expression& o);
  bool operator!=(Expression& o);
};
}
