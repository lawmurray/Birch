/**
 * @file
 */
#include "bi/visitor/DispatchGatherer.hpp"

void bi::DispatchGatherer::visit(const FuncReference* o) {
  gathered.insert(o->alternatives.begin(), o->alternatives.end());
}
