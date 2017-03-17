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
   * Is this type assignable?
   */
  bool assignable;

  /*
   * Double-dispatch partial order comparisons.
   */
  virtual bool definitely(Type& o);
  virtual bool dispatchDefinitely(Type& o) = 0;
  virtual bool definitely(AssignableType& o);
  virtual bool definitely(BracketsType& o);
  virtual bool definitely(EmptyType& o);
  virtual bool definitely(LambdaType& o);
  virtual bool definitely(List<Type>& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);
  virtual bool definitely(ParenthesesType& o);
  virtual bool definitely(RandomType& o);
  virtual bool definitely(VariantType& o);

  virtual bool possibly(Type& o);
  virtual bool dispatchPossibly(Type& o) = 0;
  virtual bool possibly(AssignableType& o);
  virtual bool possibly(BracketsType& o);
  virtual bool possibly(EmptyType& o);
  virtual bool possibly(LambdaType& o);
  virtual bool possibly(List<Type>& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(ParenthesesType& o);
  virtual bool possibly(RandomType& o);
  virtual bool possibly(VariantType& o);

  /*
   * Operators for equality comparisons.
   */
  virtual bool equals(Type& o);
};
}
