/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/statement/Class.hpp"
#include "bi/statement/AliasType.hpp"
#include "bi/statement/BasicType.hpp"

namespace bi {
/**
 * Placeholder for identifier to unknown type object.
 */
struct UnknownType {
  //
};

/**
 * Identifier for a type.
 *
 * @ingroup compiler_type
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 */
template<class ObjectType = UnknownType>
class IdentifierType: public Type,
    public Named,
    public Reference<ObjectType> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  IdentifierType(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      const bool assignable = false, const ObjectType* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  IdentifierType(const ObjectType* target);

  /**
   * Destructor.
   */
  virtual ~IdentifierType();

  virtual bool isBuiltin() const;
  virtual bool isStruct() const;
  virtual bool isClass() const;
  virtual bool isAlias() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  /**
   * Do any of the conversions for this type definitely match?
   */
  bool convertedDefinitely(const Type& o) const;

  /**
   * Do any of the conversions for this type possibly match?
   */
  bool convertedPossibly(const Type& o) const;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const IdentifierType<Class>& o) const;
  virtual bool definitely(const IdentifierType<AliasType>& o) const;
  virtual bool definitely(const IdentifierType<BasicType>& o) const;
  virtual bool definitely(const BracketsType& o) const;
  virtual bool definitely(const CoroutineType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const IdentifierType<Class>& o) const;
  virtual bool possibly(const IdentifierType<AliasType>& o) const;
  virtual bool possibly(const IdentifierType<BasicType>& o) const;
  virtual bool possibly(const BracketsType& o) const;
  virtual bool possibly(const CoroutineType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const FunctionType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
};
}
