/**
 * @file
 */
#include "bi/exception/NotFunctionException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::NotFunctionException::NotFunctionException(Call* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: call on object that is not a function\n";
  buf << "note: '" << o->single << "' is of non-function type '";
  buf << o->single->type << "'\n";
  msg = base.str();
}
