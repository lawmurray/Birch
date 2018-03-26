/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Class type.
 *
 * @ingroup type
 */
class ClassType: public Type, public Named, public Reference<Class> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeArgs Generic type arguments.
   * @param loc Location.
   * @param target Target.
   */
  ClassType(Name* name, Type* typeArgs, Location* loc = nullptr,
      Class* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   * @param loc Location.
   */
  ClassType(Class* target, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ClassType();

  virtual bool isClass() const;
  virtual Class* getClass() const;

  virtual Type* canonical();
  virtual const Type* canonical() const;

  virtual void resolveConstructor(Argumented* args);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const BasicType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const TupleType& o) const;
  virtual bool definitely(const AnyType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const BasicType& o) const;
  virtual Type* common(const ClassType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const FunctionType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const TupleType& o) const;
  virtual Type* common(const AnyType& o) const;

  /**
   * Generic type arguments.
   */
  Type* typeArgs;

  /**
   * The original target after switching to an instantiation.
   */
  Class* original;
};
}
