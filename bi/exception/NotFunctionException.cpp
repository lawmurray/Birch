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
  buf << "note: '" << o << "' is of non-function type '" << o->type << "'\n";
  msg = base.str();
}
