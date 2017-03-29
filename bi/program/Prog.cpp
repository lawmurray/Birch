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

bool bi::Prog::definitely(const Prog& o) const {
  return o.dispatchDefinitely(*this);
}

bool bi::Prog::definitely(const ProgParameter& o) const {
  return false;
}

bool bi::Prog::definitely(const ProgReference& o) const {
  return false;
}

bool bi::Prog::possibly(const Prog& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Prog::possibly(const ProgParameter& o) const {
  return false;
}

bool bi::Prog::possibly(const ProgReference& o) const {
  return false;
}
