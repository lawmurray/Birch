/**
 * @file
 */
#include "bi/exception/ConstructorException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ConstructorException::ConstructorException(const Argumented* o,
    const Class* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->args->loc) {
    buf << o->args->loc;
  }
  buf << "error: invalid call to constructor\n";
  if (o->args) {
    if (o->args->loc) {
      buf << o->args->loc;
    }
    if (o->args->isEmpty()) {
      buf << "note: no arguments\n";
    } else {
      buf << "note: argument type '" << o->args->type << "'\n";
    }
  }
  if (type) {
    if (type->loc) {
      buf << type->loc;
    }
    buf << "note: candidate\n";
    buf << type;
  }
  msg = base.str();
}
