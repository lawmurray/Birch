/**
 * @file
 */
#pragma once

#include "src/common/Located.hpp"
#include "src/expression/ExpressionIterator.hpp"
#include "src/expression/ExpressionConstIterator.hpp"

namespace birch {
class Cloner;
class Modifier;
class Visitor;

/**
 * Expression.
 *
 * @ingroup expression
 */
class Expression: public Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Expression(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param visitor The visitor.
   *
   * @return Cloned (and potentially modified) expression.
   */
  virtual Expression* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param visitor The visitor.
   *
   * @return Modified expression.
   */
  virtual Expression* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param visitor The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is expression empty?
   */
  virtual bool isEmpty() const;

  /**
   * Is result of expression assignable?
   */
  virtual bool isAssignable() const;

  /**
   * Is this a tuple expression?
   */
  virtual bool isTuple() const;

  /**
   * Is this a slice expression?
   */
  virtual bool isSlice() const;

  /**
   * Is this a membership expression?
   */
  virtual bool isMembership() const;

  /**
   * Is this a `this` expression?
   */
  virtual bool isThis() const;

  /**
   * Is this a `super` expression?
   */
  virtual bool isSuper() const;

  /**
   * Is this a reference to a global variable, function, or fiber?
   */
  virtual bool isGlobal() const;

  /**
   * Is this a reference to a member variable, function, or fiber?
   */
  virtual bool isMember() const;

  /**
   * Is this a reference to a local variable, function, or fiber?
   */
  virtual bool isLocal() const;

  /**
   * Is this a reference to a parameter?
   */
  virtual bool isParameter() const;

  /**
   * Strip parentheses, if any.
   */
  virtual const Expression* strip() const;

  /**
   * Number of items in a list.
   */
  int width() const;

  /**
   * Number of range expressions in an expression list.
   */
  virtual int depth() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  ExpressionIterator begin();

  /**
   * Iterator to one-past-the-last.
   */
  ExpressionIterator end();

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  ExpressionConstIterator begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  ExpressionConstIterator end() const;
};
}
