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

bool bi::Prog::operator<=(Prog& o) {
  return o.dispatch(*this);
}

bool bi::Prog::operator==(Prog& o) {
  return *this <= o && o <= *this;
}

bool bi::Prog::le(ProgParameter& o) {
  return false;
}

bool bi::Prog::le(ProgReference& o) {
  return false;
}
