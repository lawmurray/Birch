/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"
#include "bi/type/TypeIterator.hpp"
#include "bi/type/TypeConstIterator.hpp"

namespace bi {
class Cloner;
class Modifier;
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
   * Destructor.
   */
  virtual ~Type() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param visitor The visitor.
   *
   * @return Cloned (and potentially modified) type.
   */
  virtual Type* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param visitor The visitor.
   *
   * @return Modified type.
   */
  virtual Type* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param visitor The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is this a value type? Such a type contains no usage of class types.
   */
  bool isValue() const;

  /**
   * Is this an empty type?
   */
  virtual bool isEmpty() const;

  /**
   * Is this a basic type?
   */
  virtual bool isBasic() const;

  /**
   * Is this a class type?
   */
  virtual bool isClass() const;

  /**
   * Is this an array type?
   */
  virtual bool isArray() const;

  /**
   * Is this a list type?
   */
  virtual bool isList() const;

  /**
   * Is this a function type?
   */
  virtual bool isFunction() const;

  /**
   * Is this a fiber type?
   */
  virtual bool isFiber() const;

  /**
   * Is this a member type?
   */
  virtual bool isMember() const;

  /**
   * Is this an optional type?
   */
  virtual bool isOptional() const;

  /**
   * Is this a weak pointer type?
   */
  virtual bool isWeak() const;

  /**
   * Is this a generic type?
   */
  virtual bool isGeneric() const;

  /**
   * Get the left operand of a binary, otherwise undefined.
   */
  virtual Type* getLeft() const;

  /**
   * Get the right operand of a binary type, otherwise undefined.
   */
  virtual Type* getRight() const;

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
   * For an optional, fiber or pointer type, the type that is wrapped,
   * otherwise this.
   */
  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  /**
   * For a generic type, the argument, for an alias type, the aliased type,
   * otherwise this.
   */
  virtual Type* canonical();
  virtual const Type* canonical() const;

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
