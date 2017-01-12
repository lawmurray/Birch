/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class EmptyType;
template<class T> class List;
class ModelParameter;
class ModelReference;
class ParenthesesType;
class RandomType;

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
   * Strip parentheses.
   */
  virtual Type* strip();

  /**
   * Is this a built-in type?
   */
  virtual bool builtin() const;

  /**
   * How many dimensions does this type have?
   */
  virtual int count() const;

  /**
   * Is this type assignable?
   */
  bool assignable;

  /*
   * Partial order comparison operators for comparing types in terms of
   * specialisation. These double-dispatch to the #le, #gt, #eq and #ne
   * functions below, which can be implemented for specific types in derived
   * classes.
   */
  bool operator<=(Type& o);
  bool operator==(Type& o);
  virtual bool dispatch(Type& o) = 0;
  virtual bool le(EmptyType& o);
  virtual bool le(List<Type>& o);
  virtual bool le(ModelParameter& o);
  virtual bool le(ModelReference& o);
  virtual bool le(ParenthesesType& o);
  virtual bool le(RandomType& o);
};
}
