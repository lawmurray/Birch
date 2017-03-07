/**
 * @file
 */
#include "bi/common/Mangled.hpp"

bi::Mangled::Mangled() :
    mangled(new bi::Name()) {
  //
}

bi::Mangled::Mangled(shared_ptr<Name> mangled) :
    mangled(mangled) {
  //
}

bi::Mangled::~Mangled() {
  //
}
