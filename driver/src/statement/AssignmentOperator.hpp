/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Single.hpp"
#include "src/common/Scoped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Assignment operator.
 *
 * @ingroup statement
 */
class AssignmentOperator: public Statement,
    public Annotated,
    public Numbered,
    public Single<Expression>,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param single Operand.
   * @param braces Body.
   * @param loc Location.
   */
  AssignmentOperator(const Annotation annotation, Expression* single,
      Statement* braces, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
