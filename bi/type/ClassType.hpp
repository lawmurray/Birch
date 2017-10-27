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
 * @ingroup compiler_type
 */
class ClassType: public Type, public Named, public Reference<Class> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeArgs Generic type arguments.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  ClassType(Name* name, Type* typeArgs, Location* loc = nullptr,
      const bool assignable = false, Class* target = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  ClassType(Name* name, Location* loc = nullptr,
      const bool assignable = false, Class* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  ClassType(Class* target, Location* loc = nullptr, const bool assignable =
      false);

  /**
   * Destructor.
   */
  virtual ~ClassType();

  virtual bool isClass() const;
  virtual Class* getClass() const;

  virtual void resolveConstructor(Argumented* args);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const BasicType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const TypeList& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const TupleType& o) const;
  virtual bool definitely(const EmptyType& o) const;

  /**
   * Generic type arguments.
   */
  Type* typeArgs;
};
}
