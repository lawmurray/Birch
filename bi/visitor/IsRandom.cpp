/**
 * @file
 */
#include "bi/visitor/IsRandom.hpp"

bi::IsRandom::IsRandom() : result(false) {
  //
}

bi::IsRandom::~IsRandom() {
  //
}

void bi::IsRandom::visit(const RandomType* o) {
  result = true;
}
