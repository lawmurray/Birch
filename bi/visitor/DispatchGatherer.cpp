/**
 * @file
 */
#include "bi/visitor/DispatchGatherer.hpp"

void bi::DispatchGatherer::visit(const FuncReference* o) {
  Visitor::visit(o);
  if (o->dispatcher) {
    gathered.insert(o->dispatcher);
  }
}

void bi::DispatchGatherer::visit(const RandomInit* o) {
  Visitor::visit(o);
  o->backward->accept(this);
}
