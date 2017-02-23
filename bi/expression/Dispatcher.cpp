/**
 * @file
 */
#include "bi/expression/Dispatcher.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/visitor/all.hpp"

bi::Dispatcher::Dispatcher(shared_ptr<Name> name, shared_ptr<Name> mangled,
    shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    mangled(mangled) {
  this->arg = this;
}

bi::Dispatcher::~Dispatcher() {
  //
}

void bi::Dispatcher::insert(FuncParameter* o) {
  /* pre-condition */
  assert(*o->name == *name);
  assert(*o->mangled == *mangled);

  /* add function */
  funcs.insert(o);

  /* update possible types of parameters */
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  if (funcs.size() == 1) {
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      types.push_back(o->type.get());
    }
  } else {
    assert(gatherer.size() == types.size());
    auto iter1 = gatherer.begin();
    auto iter2 = types.begin();
    for (; iter1 != gatherer.end(); ++iter1, ++iter2) {
      auto type1 = (*iter1)->type.get();
      auto type2 = *iter2;
      VariantType* variant = dynamic_cast<VariantType*>(type2);

      if (variant) {
        /* already a variant type, add this new type to the variant */
        variant->add(type1);
      } else if (type1->definitely(*type2) || type2->definitely(*type1)) {
        /* make a new variant type */
        variant = new VariantType();
        variant->add(type1);
        variant->add(type2);
        *iter2 = variant;
        ///@todo Clean up variant on exit
      }
    }
  }
}

bi::Expression* bi::Dispatcher::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Dispatcher::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Dispatcher::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Dispatcher::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::Dispatcher::definitely(Dispatcher& o) {
  /* pre-conditions */
  assert(funcs.size() > 0);
  assert(o.funcs.size() > 0);

  /* comparing any function in one dispatcher to any function in the other
   * dispatcher is sufficient, as all should yield the same result */
  return funcs.any()->definitely(*o.funcs.any()) && o.capture(this);
}

bool bi::Dispatcher::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::Dispatcher::possibly(Dispatcher& o) {
  /* pre-conditions */
  assert(funcs.size() > 0);
  assert(o.funcs.size() > 0);

  /* comparing any function in one dispatcher to any function in the other
   * dispatcher is sufficient, as all should yield the same result */
  return funcs.any()->possibly(*o.funcs.any()) && o.capture(this);
}
