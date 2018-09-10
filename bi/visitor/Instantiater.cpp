/**
 * @file
 */
#include "bi/visitor/Instantiater.hpp"

bi::Instantiater::Instantiater(Type* typeArgs) : iter(typeArgs->begin()) {
  //
}

bi::Expression* bi::Instantiater::clone(const Generic* o) {
  auto type = (*(iter++))->canonical()->accept(this);
  return new Generic(o->name, type, o->loc);
}
