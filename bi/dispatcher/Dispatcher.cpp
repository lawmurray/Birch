/**
 * @file
 */
#include "bi/dispatcher/Dispatcher.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/visitor/all.hpp"

bi::Dispatcher::Dispatcher(shared_ptr<Name> name, shared_ptr<Name> mangled,
    Dispatcher* parent) :
    Named(name),
    Mangled(mangled),
    parent(parent) {
  //
}

bi::Dispatcher::~Dispatcher() {
  for (auto iter = paramTypes.begin(); iter != paramTypes.end(); ++iter) {
    delete *iter;
  }
}

void bi::Dispatcher::insert(FuncParameter* o) {
  /* pre-condition */
  assert(*o->name == *name);
  assert(*o->mangled == *mangled);

  /* add function */
  funcs.insert(o);
}

bi::Dispatcher* bi::Dispatcher::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Dispatcher* bi::Dispatcher::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Dispatcher::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Dispatcher::operator==(const Dispatcher& o) const {
  return false;
}
