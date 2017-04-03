/**
 * @file
 */
#include "bi/dispatcher/Dispatcher.hpp"

#include "bi/visitor/all.hpp"

#include <vector>
#include <algorithm>

bi::Dispatcher::Dispatcher(FuncReference* ref) :
    Named(ref->name), own(false) {
  /* argument types */
  std::transform(ref->parens->begin(), ref->parens->end(),
      std::back_inserter(types),
      [](const Expression* expr) {return new VariantType(expr->type.get());});

  /* result type */
  add(ref->target);
  std::for_each(ref->possibles.begin(), ref->possibles.end(),
      [&](FuncParameter* o) {add(o);});
}

bi::Dispatcher::~Dispatcher() {
  std::for_each(types.begin(), types.end(), [](VariantType* o) {delete o;});
  if (own) {
    delete type;
  }
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
  if (funcs.size() == 0) {
    type = o->result->type.get();
    own = false;
  } else {
    VariantType* variant;
    if (own) {
      variant = dynamic_cast<VariantType*>(type);
      assert(variant);
      variant->add(o->result->type.get());
    } else if (!o->result->type->equals(*type)) {
      variant = new VariantType(type);
      own = true;
      variant->add(o->result->type.get());
    }
  }
  funcs.push_back(o);

  /* post-condition */
  assert(types.size() == o->parens->tupleSize());
}
