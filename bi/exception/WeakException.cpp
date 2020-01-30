/**
 * @file
 */
#include "bi/exception/WeakException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::WeakException::WeakException(const NamedType* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: a weak reference can only be used for a class type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
