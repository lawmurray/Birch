/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * After code transformations, does some checks that are not necessarily
 * handled by the parser.
 *
 * @ingroup visitor
 */
class Checker : public Visitor {
public:
  /**
   * Constructor.
   */
  Checker();

  /**
   * Destructor.
   */
  virtual ~Checker();

  virtual void visit(const Fiber* o);
  virtual void visit(const MemberFiber* o);
  virtual void visit(const Yield* o);

private:
  /**
   * Are we in a fiber?
   */
  int inFiber;
};
}
