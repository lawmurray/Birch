/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Valued.hpp"

namespace birch {
/**
 * Declaration of multiple local variables with a tuple.
 *
 * @ingroup statement
 */
class TupleVariable: public Statement,
    public Annotated,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param locals Local variables to declare.
   * @param op Initialization operator.
   * @param value Initial value.
   * @param loc Location.
   */
  TupleVariable(const Annotation annotation, Statement* locals, Name* op,
      Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~TupleVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Local variables to declare.
   */
  Statement* locals;
};
}
