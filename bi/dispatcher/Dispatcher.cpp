/**
 * @file
 */
#include "bi/dispatcher/Dispatcher.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/visitor/all.hpp"

#include <vector>
#include <algorithm>

bi::Dispatcher::Dispatcher(shared_ptr<Name> name, shared_ptr<Name> mangled,
    Expression* parens, Dispatcher* parent) :
    Named(name),
    Mangled(mangled),
    Parenthesised(parens),
    parent(parent) {
  //
}

bi::Dispatcher::~Dispatcher() {
  //
}

void bi::Dispatcher::push_front(FuncParameter* o) {
  /* pre-condition */
  assert(*o->name == *name);
  assert(*o->mangled == *mangled);

  /* add function */
  funcs.push_front(o);
}

bool bi::Dispatcher::hasVariant() const {
  Gatherer<VarParameter> gatherer;
  parens->accept(&gatherer);
  return std::any_of(gatherer.begin(), gatherer.end(),
      [&](VarParameter* o) {return o->type->isVariant();});
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
  std::vector<FuncParameter*> funcs1, funcs2;
  std::copy(funcs.begin(), funcs.end(), std::back_inserter(funcs1));
  std::copy(o.funcs.begin(), o.funcs.end(), std::back_inserter(funcs2));
  std::sort(funcs1.begin(), funcs1.end());
  std::sort(funcs2.begin(), funcs2.end());

  return funcs1 == funcs2 && parent == o.parent && *parens == *o.parens;
}
