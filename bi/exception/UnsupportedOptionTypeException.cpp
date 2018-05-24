/**
 * @file
 */
#include "bi/exception/UnsupportedOptionTypeException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::UnsupportedOptionTypeException::UnsupportedOptionTypeException(
    Type* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (type->loc) {
    buf << type->loc;
  }
  buf << "error: unsupported type '" << type << "' for program option, supported types are Boolean, Integer, Real and String\n";
  msg = base.str();
}
