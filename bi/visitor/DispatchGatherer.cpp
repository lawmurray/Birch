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
}

void bi::DispatchGatherer::insert(const FuncParameter* o) {
  gathered.insert(o);
  std::list<FuncParameter*> parents;
  scope->parents(const_cast<FuncParameter*>(o), parents);
  for (auto iter = parents.begin(); iter != parents.end(); ++iter) {
    insert(*iter);
  }
}
