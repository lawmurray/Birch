/**
 * @file
 */
#include "bi/type/Type.hpp"

bi::Type::Type(shared_ptr<Location> loc) :
    Located(loc),
    assignable(false) {
  //
}
