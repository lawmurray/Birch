/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class AliasType;
class ArrayType;
class BasicType;
class ClassType;
class CoroutineType;
class EmptyType;
class FunctionType;
class IdentifierType;
template<class T> class Iterator;
template<class T> class List;
class ParenthesesType;

class Class;
class Alias;
class Basic;

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
   * @param assignable Is this type assignable?
   */
  Type(shared_ptr<Location> loc = nullptr, const bool assignable = false);

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
  virtual Type* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified type.
   */
  virtual Type* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is this an empty type?
   */
  virtual bool isEmpty() const;

  /**
   * Is this a built-in type?
   */
  virtual bool isBasic() const;

  /**
   * Is this a class type, or an alias of a class type?
   */
  virtual bool isClass() const;

  /**
   * Is this an type?
   */
  virtual bool isAlias() const;

  /**
   * Is this an array type?
   */
  virtual bool isArray() const;

  /**
   * Is this a function type?
   */
  virtual bool isFunction() const;

  /**
   * Is this a coroutine type?
   */
  virtual bool isCoroutine() const;

  /**
   * Strip parentheses.
   */
  virtual Type* strip();

  /**
   * How many dimensions does this type have?
   */
  virtual int count() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  Iterator<Type> begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  Iterator<Type> end() const;

  /**
   * Is this type assignable?
   */
  bool assignable;

  /*
   * Double-dispatch partial order comparisons.
   */
  virtual bool definitely(const Type& o) const;
  virtual bool dispatchDefinitely(const Type& o) const = 0;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const BasicType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const CoroutineType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool possibly(const Type& o) const;
  virtual bool dispatchPossibly(const Type& o) const = 0;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const ArrayType& o) const;
  virtual bool possibly(const BasicType& o) const;
  virtual bool possibly(const ClassType& o) const;
  virtual bool possibly(const CoroutineType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const FunctionType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ParenthesesType& o) const;

  /**
   * Are these two types the same?
   */
  virtual bool equals(const Type& o) const;
};
}
