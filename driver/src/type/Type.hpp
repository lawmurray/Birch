/**
 * @file
 */
#pragma once

#include "src/common/Located.hpp"
#include "src/type/TypeIterator.hpp"
#include "src/type/TypeConstIterator.hpp"

namespace birch {
class Visitor;

/**
 * Type.
 *
 * @ingroup type
 */
class Type: public Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Type(Location* loc = nullptr);

  /**
   * Accept read-only visitor.
   *
   * @param visitor The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is this an empty type?
   */
  virtual bool isEmpty() const;

  /**
   * Is this an array type?
   */
  virtual bool isArray() const;

  /**
   * Is this a list type?
   */
  virtual bool isTuple() const;

  /**
   * Is this a member type?
   */
  virtual bool isMember() const;

  /**
   * Is this an optional type?
   */
  virtual bool isOptional() const;
  
  /**
   * Is this a future type?
   */
  virtual bool isFuture() const;

  /**
   * Is this a deduced type?
   */
  virtual bool isDeduced() const;

  /**
   * Number of elements in a type list.
   */
  virtual int width() const;

  /**
   * For an array type, the number of dimensions, for a sequence type, the
   * number of nested sequences, otherwise zero.
   */
  virtual int depth() const;

  /**
   * For an optional or pointer type, the type that is wrapped, otherwise
   * this.
   */
  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  /**
   * For a sequence or array type, the element type, otherwise this.
   */
  virtual Type* element();
  virtual const Type* element() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  TypeIterator begin();
  TypeConstIterator begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  TypeIterator end();
  TypeConstIterator end() const;
};
}
