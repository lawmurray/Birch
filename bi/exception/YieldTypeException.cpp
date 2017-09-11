/**
 * @file
 */
#include "bi/exception/YieldTypeException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::YieldTypeException::YieldTypeException(const Yield* o, const Type* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: yield value has incorrect type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  buf << "note: value has type '" << o->single->type << "', should be '" << type << "'\n";
  msg = base.str();
}
