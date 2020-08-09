/**
 * @file
 */
#include "bi/exception/SpinException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::SpinException::SpinException(const Spin* o) {
  std::stringstream base;
  bih_ostream buf(base);
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
