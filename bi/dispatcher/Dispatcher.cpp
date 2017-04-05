/**
 * @file
 */
#include "bi/dispatcher/Dispatcher.hpp"

#include "bi/visitor/all.hpp"

#include <vector>
#include <algorithm>

bi::Dispatcher::Dispatcher(shared_ptr<Name> name) :
    Named(name) {
  //
}

bi::Dispatcher::~Dispatcher() {
  std::for_each(types.begin(), types.end(), [](VariantType* o) {delete o;});
}

bi::Dispatcher* bi::Dispatcher::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Dispatcher::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Dispatcher::operator==(const Dispatcher& o) const {
  std::vector<FuncParameter*> funcs1, funcs2;
  std::copy(funcs.begin(), funcs.end(), std::back_inserter(funcs1));
  std::copy(o.funcs.begin(), o.funcs.end(), std::back_inserter(funcs2));
  std::sort(funcs1.begin(), funcs1.end());
  std::sort(funcs2.begin(), funcs2.end());

  return funcs1 == funcs2;
}

void bi::Dispatcher::add(FuncParameter* o) {
  funcs.push_back(o);
}
