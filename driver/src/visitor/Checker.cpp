/**
 * @file
 */
#include "src/visitor/Checker.hpp"

#include "src/exception/all.hpp"

birch::Checker::Checker() : inFiber(0), inLambda(0) {
  //
}

birch::Checker::~Checker() {
  //
}

void birch::Checker::visit(const LambdaFunction* o) {
  ++inLambda;
  Visitor::visit(o);
  --inLambda;
}

void birch::Checker::visit(const Fiber* o) {
  ++inFiber;
  Visitor::visit(o);
  --inFiber;
}

void birch::Checker::visit(const MemberFiber* o) {
  ++inFiber;
  Visitor::visit(o);
  --inFiber;
}

void birch::Checker::visit(const Yield* o) {
  if (!inFiber || inLambda) {
    throw YieldException(o);
  }
  Visitor::visit(o);
}
