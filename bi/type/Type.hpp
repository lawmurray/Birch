/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class AssignableType;
class BracketsType;
class EmptyType;
template<class T> class Iterator;
class LambdaType;
template<class T> class List;
class ModelParameter;
class ModelReference;
class ParenthesesType;
class RandomType;
class VariantType;

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
   * @param assignable Is this type writeable?
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
  virtual bool isBuiltin() const;

  /**
   * Is this a model type?
   */
  virtual bool isModel() const;

  /**
   * Is this a random type?
   */
  virtual bool isRandom() const;

  /**
   * Is this a lambda type?
   */
  virtual bool isLambda() const;

  /**
   * Is this a variant type?
   */
  virtual bool isVariant() const;

  /**
   * Strip parentheses.
   */
  virtual Type* strip();

  /**
   * How many dimensions does this type have?
   */
  virtual int count() const;

  /**
   * Iterator to first element if this is an list, otherwise to this.
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
  virtual bool definitely(const AssignableType& o) const;
  virtual bool definitely(const BracketsType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const LambdaType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ModelParameter& o) const;
  virtual bool definitely(const ModelReference& o) const;
  virtual bool definitely(const ParenthesesType& o) const;
  virtual bool definitely(const RandomType& o) const;
  virtual bool definitely(const VariantType& o) const;

  virtual bool possibly(const Type& o) const;
  virtual bool dispatchPossibly(const Type& o) const = 0;
  virtual bool possibly(const AssignableType& o) const;
  virtual bool possibly(const BracketsType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const LambdaType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ModelParameter& o) const;
  virtual bool possibly(const ModelReference& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
  virtual bool possibly(const RandomType& o) const;
  virtual bool possibly(const VariantType& o) const;

  /*
   * Operators for equality comparisons.
   */
  virtual bool equals(Type& o);
};
}
