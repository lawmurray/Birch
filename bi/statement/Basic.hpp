/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/TypeParameterised.hpp"
#include "bi/common/Based.hpp"

namespace bi {
/**
 * Basic type.
 *
 * @ingroup statement
 */
class Basic: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Based {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param base Base type.
   * @param alias Is this an alias relationship?
   * @param loc Location.
   */
  Basic(const Annotation annotation, Name* name, Expression* typeParams,
      Type* base, const bool alias, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Basic();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
