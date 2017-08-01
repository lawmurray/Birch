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
  virtual Iterator<Expression> begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  virtual Iterator<Expression> end() const;
};
}
