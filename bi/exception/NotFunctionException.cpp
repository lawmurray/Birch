/**
 * @file
 */
#include "bi/exception/NotFunctionException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::NotFunctionException::NotFunctionException(Expression* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: not a function\n";
  buf << "note: '" << o << "' is not a function\n";
  msg = base.str();
}
