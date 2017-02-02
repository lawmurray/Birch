/**
 * @file
 */
#include "bi/visitor/DispatchGatherer.hpp"

bi::DispatchGatherer::DispatchGatherer(Scope* scope) :
    scope(scope) {
  //
}

void bi::DispatchGatherer::visit(const FuncReference* o) {
  for (auto iter = o->alternatives.begin(); iter != o->alternatives.end();
      ++iter) {
    insert(*iter);
  }
  if (o->alternatives.size() > 0 && o->target) {
    insert(o->target);
  }
}

void bi::DispatchGatherer::visit(const RandomInit* o) {
  Visitor::visit(o);
  o->push->accept(this);
}

void bi::DispatchGatherer::insert(const FuncParameter* o) {
  gathered.insert(o);
  std::list<FuncParameter*> parents;
  scope->parents(const_cast<FuncParameter*>(o), parents);
  for (auto iter = parents.begin(); iter != parents.end(); ++iter) {
    insert(*iter);
  }

  /*
   * Also need any functions used in the signatures of imported
   * functions. Note that as the bodies of imported functions are not
   * resolved, this won't introduce any extraneous functions that are used
   * in the bodies, rather than just the signatures, of imported functions.
   */
  o->accept(this);
}
