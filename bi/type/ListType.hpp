/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Iterator.hpp"

namespace bi {
/**
 * List type.
 *
 * @ingroup compiler_common
 */
class ListType: public Type {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  ListType(Type* head, Type* tail, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~ListType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isList() const;

  /**
   * Left operand.
   */
  Type* head;

  /**
   * Right operand.
   */
  Type* tail;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const ListType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const ListType& o) const;
  virtual bool possibly(const OptionalType& o) const;
};
}
