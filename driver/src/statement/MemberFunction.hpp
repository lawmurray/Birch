/**
 * @file
 */
#pragma once

#include "src/statement/Function.hpp"

namespace birch {
/**
 * Member function.
 *
 * @ingroup statement
 */
class MemberFunction: public Function {
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
  MemberFunction(const Annotation annotation, Name* name,
      Expression* typeParams, Expression* params, Type* returnType,
      Statement* braces, Location* loc = nullptr);

  virtual bool isMember() const;

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
