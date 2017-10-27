/**
 * @file
 */
#include "bi/exception/GenericException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::GenericException::GenericException(const ClassType* ref, const Class* param) {
  std::stringstream base;
  bih_ostream buf(base);
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "error: invalid generic type arguments\n";

  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "note: in\n";
  buf << ref << '\n';

  if (param->loc) {
    buf << param->loc;
  }
  buf << "note: class is\n";
  buf << param->name << '<' << param->typeParams << ">\n";
  msg = base.str();
}
