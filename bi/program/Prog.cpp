/**
 * @file
 */
#include "bi/program/Prog.hpp"

bi::Prog::Prog(shared_ptr<Location> loc) :
    Located(loc) {
  //
}

bi::Prog::~Prog() {
  //
}

bi::possibly bi::Prog::operator<=(Prog& o) {
  return o.dispatch(*this);
}

bi::possibly bi::Prog::operator==(Prog& o) {
  return *this <= o && o <= *this;
}

bi::possibly bi::Prog::le(ProgParameter& o) {
  return untrue;
}

bi::possibly bi::Prog::le(ProgReference& o) {
  return untrue;
}
