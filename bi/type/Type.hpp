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

class Argumented;

class ArrayType;
class BasicType;
class BinaryType;
class ClassType;
class FiberType;
class EmptyType;
class FunctionType;
class GenericType;
class MemberType;
class NilType;
class OptionalType;
class OverloadedType;
class PointerType;
class SequenceType;
class TupleType;
class TypeConstIterator;
class UnknownType;
class TypeList;

class Class;
class Basic;

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
   * Is this a value type?
   */
  virtual bool isValue() const;

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
   * Is this a sequence type?
   */
  virtual bool isSequence() const;

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
   * Is this a pointer type?
   */
  virtual bool isPointer() const;

  /**
   * Is this a weak pointer type?
   */
  virtual bool isWeak() const;

  /**
   * Is this a binary operator type?
   */
  virtual bool isBinary() const;

  /**
   * Is this an overloaded type?
   */
  virtual bool isOverloaded() const;

  /**
   * Are all the generics of this class defined?
   */
  virtual bool isBound() const;

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
   * Resolve a call without arguments.
   *
   * @return If this is an overloaded type, but there is only one overload,
   * return the type of that overload. In all other cases undefined.
   */
  virtual FunctionType* resolve() const;

  /**
   * Resolve a constructor call.
   *
   * @param args Argument types.
   */
  virtual void resolveConstructor(Argumented* o);

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

  /**
   * Is this type equal to another?
   */
  virtual bool equals(const Type& o) const;

  /*
   * Double-dispatch partial order comparisons.
   */
  virtual bool definitely(const Type& o) const;
  virtual bool dispatchDefinitely(const Type& o) const = 0;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const BasicType& o) const;
  virtual bool definitely(const BinaryType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const MemberType& o) const;
  virtual bool definitely(const NilType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const OverloadedType& o) const;
  virtual bool definitely(const PointerType& o) const;
  virtual bool definitely(const SequenceType& o) const;
  virtual bool definitely(const TupleType& o) const;
  virtual bool definitely(const UnknownType& o) const;
  virtual bool definitely(const TypeList& o) const;

  /*
   * Double-dispatch common type reductions.
   */
  virtual Type* common(const Type& o) const;
  virtual Type* dispatchCommon(const Type& o) const = 0;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const BasicType& o) const;
  virtual Type* common(const BinaryType& o) const;
  virtual Type* common(const ClassType& o) const;
  virtual Type* common(const EmptyType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const FunctionType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const NilType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const OverloadedType& o) const;
  virtual Type* common(const PointerType& o) const;
  virtual Type* common(const SequenceType& o) const;
  virtual Type* common(const TupleType& o) const;
  virtual Type* common(const UnknownType& o) const;
  virtual Type* common(const TypeList& o) const;
};
}
