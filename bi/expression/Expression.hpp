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

class BracketsExpression;
class EmptyExpression;
class Index;
template<class T> class Identifier;
template<class T> class Iterator;
class LambdaFunction;
template<class T> class List;
template<class T> class Literal;
class Member;
class Parameter;
class ParenthesesExpression;
class Range;
class Span;
class Super;
class This;
class Unknown;

class GlobalVariable;
class LocalVariable;
class MemberVariable;
class Function;
class Coroutine;
class MemberFunction;
class BinaryOperator;
class UnaryOperator;

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

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  Iterator<Expression> begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  Iterator<Expression> end() const;

  /*
   * Double-dispatch partial order comparisons.
   */
  bool definitely(const Expression& o) const;
  virtual bool dispatchDefinitely(const Expression& o) const = 0;
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
  virtual bool definitely(const Literal<bool>& o) const;
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

  bool possibly(const Expression& o) const;
  virtual bool dispatchPossibly(const Expression& o) const = 0;
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
  virtual bool possibly(const Literal<bool>& o) const;
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

  /**
   * Operators for equality comparisons.
   */
  bool operator==(Expression& o);
  bool operator!=(Expression& o);
};
}
