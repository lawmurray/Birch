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
  /* check that these dispatchers contain the same functions */
  std::vector<FuncParameter*> funcs1, funcs2;
  std::copy(funcs.begin(), funcs.end(), std::back_inserter(funcs1));
  std::copy(o.funcs.begin(), o.funcs.end(), std::back_inserter(funcs2));
  std::sort(funcs1.begin(), funcs1.end());
  std::sort(funcs2.begin(), funcs2.end());

  /* check whether the parameter types are the same */
  auto iter1 = types.begin();
  auto end1 = types.end();
  auto iter2 = o.types.begin();
  auto end2 = o.types.end();
  while (iter1 != end1 && iter2 != end2 && (*iter1)->equals(**iter2)) {
    ++iter1;
    ++iter2;
  }
  return funcs1 == funcs2 && iter1 == end1 && iter2 == end2;
}

void bi::Dispatcher::add(FuncParameter* o) {
  funcs.push_back(o);
}
