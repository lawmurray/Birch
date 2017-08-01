/**
 * @file
 */
#include "bi/common/Named.hpp"

bi::Named::Named() :
    name(new bi::Name()) {
  //
}

bi::Named::Named(Name* name) :
    name(name) {
  //
}

bi::Named::~Named() {
  //
}
