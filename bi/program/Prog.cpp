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

bool bi::Prog::definitely(Prog& o) {
  return o.dispatchDefinitely(*this);
}

bool bi::Prog::definitely(ProgParameter& o) {
  return false;
}

bool bi::Prog::definitely(ProgReference& o) {
  return false;
}

bool bi::Prog::possibly(Prog& o) {
  return o.dispatchPossibly(*this);
}

bool bi::Prog::possibly(ProgParameter& o) {
  return false;
}

bool bi::Prog::possibly(ProgReference& o) {
  return false;
}
