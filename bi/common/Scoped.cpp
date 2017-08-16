/**
 * @file
 */
#include "bi/common/Scoped.hpp"

bi::Scoped::Scoped() : scope(new Scope()) {
  //
}

bi::Scoped::~Scoped() {
  //
}
