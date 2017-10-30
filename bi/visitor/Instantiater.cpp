/**
 * @file
 */
#include "bi/visitor/Instantiater.hpp"

bi::Instantiater::Instantiater(Type* typeArgs) : iter(typeArgs->begin()) {
  //
}

bi::Expression* bi::Instantiater::clone(const Generic* o) {
  return new Generic(o->name, (*(iter++))->accept(this), o->loc);
}
