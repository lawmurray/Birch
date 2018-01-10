/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"

namespace bi {
/**
 * Unknown type at parse time. This is used as a placeholder by the parser
 * for types that cannot be resolved syntactically, and must be deferred to
 * resolution passes.
 *
 * @ingroup birch_type
 */
class UnknownType: public Type, public Named {
public:
  /**
   * Constructor.
   *
   * @param weak Does this have a weak marker?
   * @param name Name.
   * @param typeArgs Generic type arguments.
   * @param read Does this have a read-only marker?
   * @param loc Location.
   */
  UnknownType(const bool weak, Name* name, Type* typeArgs,
      const bool read, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnknownType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual Type* dispatchCommon(const Type& o) const;

  /**
   * Does this have a weak marker?
   */
  bool weak;

  /**
   * Generic type arguments.
   */
  Type* typeArgs;

  /**
   * Does this have a read-only marker?
   */
  bool read;
};
}
