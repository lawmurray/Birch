/**
 * @file
 */
#pragma once

#include "bi/common/Typed.hpp"
#include "bi/common/Located.hpp"
#include "bi/expression/ExpressionIterator.hpp"
#include "bi/expression/ExpressionConstIterator.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

/**
 * Expression.
 *
 * @ingroup expression
 */
class Expression: public Located, public Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   * @param loc Location.
   */
  Expression(Type* type, Location* loc = nullptr);

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
   * Is this a value expression? Such an expression contains no usage of
   * class types.
   */
  bool isValue() const;

  /**
   * Is expression empty?
   */
  virtual bool isEmpty() const;

  /**
   * Is result of expression assignable?
   */
  virtual bool isAssignable() const;

  /**
   * Strip parentheses, if any.
   */
  virtual Expression* strip();

  /**
   * Get the left operand of a binary, otherwise undefined.
   */
  virtual Expression* getLeft() const;

  /**
   * Get the right operand of a binary, otherwise undefined.
   */
  virtual Expression* getRight() const;

  /**
   * Number of expresions in an expression list.
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
