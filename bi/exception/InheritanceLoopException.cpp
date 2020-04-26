/**
 * @file
 */
#include "bi/exception/InheritanceLoopException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::InheritanceLoopException::InheritanceLoopException(const Class* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: loop in class inheritance.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
