/**
 * @file
 */
#include "bi/common/Named.hpp"

bi::Named::Named() :
    name(new bi::Name()) {
  //
}

bi::Named::Named(shared_ptr<Name> name) :
    name(name) {
  //
}

bi::Named::~Named() {
  //
}
