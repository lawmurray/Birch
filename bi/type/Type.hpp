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
 * Type.
 *
 * @ingroup compiler_type
 */
class Type: public Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Type(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Type() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) type.
   */
  virtual Type* acceptClone(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified type.
   */
  virtual Type* acceptModify(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Bool cast to check for non-empty Type.
   */
  virtual operator bool() const;

  /**
   * Is this a built-in type?
   */
  virtual bool builtin() const;

  /**
   * How many dimensions does this type have?
   */
  virtual int count() const;

  /*
   * Partial order comparison operators for comparing types in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(Type& o) = 0;
  virtual bool operator==(const Type& o) const = 0;
  bool operator<(const Type& o) const;
  bool operator>(const Type& o) const;
  bool operator>=(const Type& o) const;
  bool operator!=(const Type& o) const;

  /**
   * Is this type assignable?
   */
  bool assignable;
};
}
