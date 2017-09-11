/**
 * @file
 */
#include "bi/exception/ReturnTypeException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ReturnTypeException::ReturnTypeException(const Return* o, const Type* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: return value has incorrect type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  buf << "note: value has type '" << o->single->type << "', should be '" << type << "'\n";
  msg = base.str();
}
