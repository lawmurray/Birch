/**
 * @file
 */
#include "bi/expression/Dispatcher.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/visitor/all.hpp"

bi::Dispatcher::Dispatcher(shared_ptr<Name> name, Expression* parens,
    Expression* result, shared_ptr<Location> loc) :
    Expression(loc),
    Signature(name, parens, result) {
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

  /* update variant types */
  update(parens.get(), o->parens.get());
  update(result.get(), o->result.get());
}

void bi::Dispatcher::update(Expression* o1, Expression* o2) {
  assert(o1->possibly(*o2));
  if (o1->possibly(*o2)) {  // capture arguments
    Cloner cloner;
    Gatherer<VarParameter> gatherer;

    o2->accept(&gatherer);
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = const_cast<VarParameter*>(*iter);
      VariantType* variant = dynamic_cast<VariantType*>(param->type.get());
      if (variant) {
        /* this parameter already has a variant type, add the type of the
         * argument to this */
        variant->add(param->arg->type->accept(&cloner));
      } else if (*param->type != *param->arg->type) {
        /* make a new variant type */
        variant = new VariantType();
        variant->add(param->type.release());
        variant->add(param->arg->type->accept(&cloner));
        param->type = variant;
      }
    }
  }
}

bi::Expression * bi::Dispatcher::accept(Cloner * visitor) const {
  return visitor->clone(this);
}

bi::Expression * bi::Dispatcher::accept(Modifier * visitor) {
  return visitor->modify(this);
}

void bi::Dispatcher::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Dispatcher::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::Dispatcher::definitely(Dispatcher& o) {
  return parens->definitely(*o.parens) && o.capture(this);
}

bool bi::Dispatcher::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::Dispatcher::possibly(Dispatcher& o) {
  return parens->possibly(*o.parens) && o.capture(this);
}

bool bi::Dispatcher::possibly(FuncParameter& o) {
  return parens->possibly(*o.parens) && o.capture(this);
}
