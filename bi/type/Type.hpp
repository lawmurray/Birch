/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class Argumented;

class AliasType;
class ArrayType;
class BasicType;
class BinaryType;
class ClassType;
class FiberType;
class EmptyType;
class FunctionType;
class GenericType;
class NilType;
class OptionalType;
class OverloadedType;
class TupleType;
class TypeIdentifier;
class TypeIterator;
class TypeList;

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
  Type(Location* loc = nullptr, const bool assignable = false);

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
   * Is this an optional type?
   */
  virtual bool isOptional() const;

  /**
   * Is this a binary operator type?
   */
  virtual bool isBinary() const;

  /**
   * Is this an overloaded type?
   */
  virtual bool isOverloaded() const;

  /**
   * Get the left operand of a binary, otherwise undefined.
   */
  virtual Type* getLeft() const;

  /**
   * Get the right operand of a binary type, otherwise undefined.
   */
  virtual Type* getRight() const;

  /**
   * Get the statement associated with a class type, otherwise undefined.
   */
  virtual Class* getClass() const;

  /**
   * Get the statement associated with a basic type, otherwise undefined.
   */
  virtual Basic* getBasic() const;

  /**
   * For an optional or fiber type, the type that is wrapped, otherwise
   * undefined.
   */
  virtual Type* unwrap() const;

  /**
   * Resolve a call.
   *
   * @param args Arguments.
   *
   * @return If this is an overloaded type, and the argument types match the
   * parameter types of the function, the return type of the function. In all
   * other cases throws an exception.
   */
  virtual FunctionType* resolve(Argumented* o);

  /**
   * Resolve a constructor call.
   *
   * @param args Argument types.
   */
  virtual void resolveConstructor(Argumented* o);

  /**
   * How many dimensions does this type have?
   */
  virtual int count() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  TypeIterator begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  TypeIterator end() const;

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
  virtual bool definitely(const BinaryType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const NilType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const OverloadedType& o) const;
  virtual bool definitely(const TupleType& o) const;
  virtual bool definitely(const TypeIdentifier& o) const;
  virtual bool definitely(const TypeList& o) const;

  /**
   * Are these two types the same?
   */
  virtual bool equals(const Type& o) const;
};
}
