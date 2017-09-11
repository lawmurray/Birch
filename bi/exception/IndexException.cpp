/**
 * @file
 */
#include "bi/exception/IndexException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::IndexException::IndexException(const Expression* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: incompatible type in index";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  buf << "note: index has type '" << o->type << "', but must be integer\n";
  msg = base.str();
}
