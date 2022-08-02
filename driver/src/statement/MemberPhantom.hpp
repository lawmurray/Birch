/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"

namespace birch {
/**
 * Phantom class member variable. This is a variable that is declared in
 * nested C++ code but that should be included in the boilerplate necessary
 * for sweeping through objects for copy-on-write.
 *
 * @ingroup statement
 */
class MemberPhantom: public Statement,
    public Annotated,
    public Named,
    public Numbered {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param loc Location.
   */
  MemberPhantom(const Annotation annotation, Name* name,
      Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
