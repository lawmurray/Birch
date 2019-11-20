/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"

namespace bi {
/**
 * Index variable in a parallel loop.
 *
 * @ingroup expression
 */
class ParallelVariable: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public Typed {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  ParallelVariable(const Annotation annotation, Name* name, Type* type,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ParallelVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
