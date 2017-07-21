/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/statement/Alias.hpp"

namespace bi {
/**
 * Alias type.
 *
 * @ingroup compiler_type
 */
class AliasType: public Type,
    public Named,
    public Reference<Alias> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  AliasType(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      const bool assignable = false, const Alias* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  AliasType(const Alias* target);

  /**
   * Destructor.
   */
  virtual ~AliasType();

  virtual bool isBasic() const;
  virtual bool isClass() const;
  virtual bool isAlias() const;
  virtual bool isArray() const;
  virtual bool isFunction() const;
  virtual bool isCoroutine() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const BasicType& o) const;
  virtual bool definitely(const ClassType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ParenthesesType& o) const;
  virtual bool definitely(const Alias& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const ArrayType& o) const;
  virtual bool possibly(const BasicType& o) const;
  virtual bool possibly(const ClassType& o) const;
  virtual bool possibly(const FiberType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const FunctionType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
};
}
