/**
 * @file
 */
#include "bi/visitor/Checker.hpp"

#include "bi/exception/all.hpp"

bi::Checker::Checker() : inFiber(0), inLambda(0) {
  //
}

bi::Checker::~Checker() {
  //
}

void bi::Checker::visit(const LambdaFunction* o) {
  ++inLambda;
  Visitor::visit(o);
  --inLambda;
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
  if (!inFiber || inLambda) {
    throw YieldException(o);
  }
  Visitor::visit(o);
}
