/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

/**
 * Statement.
 *
 * @ingroup compiler_statement
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
};
}
