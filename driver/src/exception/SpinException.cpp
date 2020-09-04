/**
 * @file
 */
#include "src/exception/SpinException.hpp"

#include "src/generate/BirchGenerator.hpp"

birch::SpinException::SpinException(const Spin* o) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: spin outside fiber.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
