/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * List type.
 *
 * @ingroup compiler_common
 */
class TypeList: public Type {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   */
  TypeList(Type* head, Type* tail, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~TypeList();

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

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const TypeList& o) const;
};
}
