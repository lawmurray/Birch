/**
 * @file
 */
#pragma once

#include "bi/statement/Fiber.hpp"

namespace bi {
/**
 * Member fiber.
 *
 * @ingroup statement
 */
class MemberFiber: public Fiber {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param typeParams Type parameters.
   * @param params Parameters.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  MemberFiber(const Annotation annotation, Name* name, Expression* typeParams,
      Expression* params, Type* returnType, Statement* braces,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberFiber();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
