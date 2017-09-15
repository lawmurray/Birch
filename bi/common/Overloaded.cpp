/**
 * @file
 */
#include "bi/common/Overloaded.hpp"

#include "bi/exception/all.hpp"

bi::Overloaded::Overloaded(Parameterised* o) {
  add(o);
}

bool bi::Overloaded::contains(Parameterised* o) {
  return overloads.contains(o);
}

void bi::Overloaded::add(Parameterised* o) {
  /* pre-condition */
  assert(!contains(o));

  overloads.insert(o);
}
