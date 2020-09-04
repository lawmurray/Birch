/**
 * @file
 */
#include "src/common/Named.hpp"

birch::Named::Named() :
    name(new birch::Name()) {
  //
}

birch::Named::Named(Name* name) :
    name(name) {
  //
}

birch::Named::~Named() {
  //
}
