/**
 * @file
 */
#include "bi/exception/YieldException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::YieldException::YieldException(const Yield* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: yield outside fiber.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
