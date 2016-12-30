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

bi::Prog::operator bool() const {
  return true;
}

bool bi::Prog::operator<(Prog& o) {
  return *this <= o && *this != o;
}

bool bi::Prog::operator>(Prog& o) {
  return o <= *this && o != *this;
}

bool bi::Prog::operator>=(Prog& o) {
  return o <= *this;
}

bool bi::Prog::operator!=(Prog& o) {
  return !(*this == o);
}
