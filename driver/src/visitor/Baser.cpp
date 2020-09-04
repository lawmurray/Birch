/**
 * @file
 */
#include "src/visitor/Baser.hpp"

#include "src/exception/all.hpp"

birch::Baser::Baser() {
  //
}

birch::Baser::~Baser() {
  //
}

birch::Statement* birch::Baser::modify(Class* o) {
  scopes.back()->inherit(o);
  return Modifier::modify(o);
}
