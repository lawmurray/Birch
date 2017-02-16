/**
 * @file
 */
#include "bi/visitor/DispatchGatherer.hpp"

void bi::DispatchGatherer::visit(const FuncReference* o) {
  Visitor::visit(o);
  for (auto iter = o->alternatives.begin(); iter != o->alternatives.end();
      ++iter) {
    insert(*iter);
  }
  if (o->alternatives.size() > 0) {
    insert(o->target);
  }
}

void bi::DispatchGatherer::visit(const RandomInit* o) {
  Visitor::visit(o);
  o->backward->accept(this);
}

void bi::DispatchGatherer::insert(const FuncParameter* o) {
  if (gathered.insert(o).second) {
    Visitor::visit(o);

    /*
     * Also need any functions used in the signatures of imported
     * functions. Note that as the bodies of imported functions are not
     * resolved, this won't introduce any extraneous functions that are used
     * in the bodies, rather than just the signatures, of imported functions.
     */
    o->accept(this);
  }
}
