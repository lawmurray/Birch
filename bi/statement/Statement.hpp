/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;
class StatementIterator;

/**
 * Statement.
 *
 * @ingroup birch_statement
 */
class Statement: public Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Statement(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Statement() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) statement.
   */
  virtual Statement* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified statement.
   */
  virtual Statement* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Strip braces, if any.
   */
  virtual Statement* strip();

  /*
   * Is statement empty?
   */
  virtual bool isEmpty() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  StatementIterator begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  StatementIterator end() const;

  /**
   * Number of elements when iterating.
   */
  int count() const;
};
}
