/**
 * @file
 */
#include "bi/exception/InvalidCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::InvalidCallException::InvalidCallException(Call* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid call '" << o << "'\n";
  if (o->single->type->isFunction()) {
    buf << "note: function type is\n";
    buf << o->single->type << "\n";
    buf << "note: argument type is\n";
    buf << o->args->type << "\n";
  } else {
    buf << "note: expression is not of function type:\n";
    buf << o->single->type << "\n";
  }
  msg = base.str();
}
