/**
 * @file
 */
#pragma once

#include "src/common/Located.hpp"
#include "src/expression/ExpressionIterator.hpp"
#include "src/expression/ExpressionConstIterator.hpp"

namespace birch {
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
