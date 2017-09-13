/**
 * @file
 */
#include "bi/exception/ConstructorException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ConstructorException::ConstructorException(const Type* args,
    const Class* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (args->loc) {
    buf << args->loc;
  }
  buf << "error: invalid call to constructor\n";
  if (args) {
    if (args->loc) {
      buf << args->loc;
    }
    if (args->isEmpty()) {
      buf << "note: no arguments\n";
    } else {
      buf << "note: argument type '" << args << "'\n";
    }
  }
  if (type) {
    if (type->loc) {
      buf << type->loc;
    }
    if (type->parens->type->isEmpty()) {
      buf << "note: no parameters\n";
    } else {
      buf << "note: parameter type '" << type->parens->type << "'\n";
    }
  }
  msg = base.str();
}
