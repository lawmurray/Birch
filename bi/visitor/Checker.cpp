/**
 * @file
 */
#include "bi/visitor/Checker.hpp"

#include "bi/exception/all.hpp"

bi::Checker::Checker() : inFiber(0) {
  //
}

bi::Checker::~Checker() {
  //
}

void bi::Checker::visit(const Fiber* o) {
  ++inFiber;
  Visitor::visit(o);
  --inFiber;
}

void bi::Checker::visit(const MemberFiber* o) {
  ++inFiber;
  Visitor::visit(o);
  --inFiber;
}

void bi::Checker::visit(const Yield* o) {
  if (!inFiber) {
    throw YieldException(o);
  }
  Visitor::visit(o);
}
