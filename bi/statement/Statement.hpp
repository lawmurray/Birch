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
  Statement(shared_ptr<Location> loc = nullptr);

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
  virtual Statement* acceptClone(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified statement.
   */
  virtual Statement* acceptModify(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /*
   * Bool cast to check for non-empty statement.
   */
  virtual operator bool() const;

  /*
   * Partial order comparison operators for comparing statements in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(Statement& o) = 0;
  virtual bool operator==(const Statement& o) const = 0;
  bool operator<(Statement& o);
  bool operator>(Statement& o);
  bool operator>=(Statement& o);
  bool operator!=(Statement& o);
};
}
