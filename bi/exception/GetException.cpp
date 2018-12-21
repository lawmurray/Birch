/**
 * @file
 */
#include "bi/exception/GetException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::GetException::GetException(const Get* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: postfix '!' can only be used with an optional, fiber, or weak type\n";
  buf << "note: operand has type '" << o->single->type << "'\n";
  msg = base.str();
}
