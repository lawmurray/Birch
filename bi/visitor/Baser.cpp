/**
 * @file
 */
#include "bi/visitor/Baser.hpp"

#include "bi/exception/all.hpp"

bi::Baser::Baser() {
  //
}

bi::Baser::~Baser() {
  //
}

bi::Statement* bi::Baser::modify(Class* o) {
  scopes.back()->inherit(o);
  return Modifier::modify(o);
}
