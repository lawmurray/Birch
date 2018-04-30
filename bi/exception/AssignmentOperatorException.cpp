/**
 * @file
 */
#include "bi/exception/AssignmentOperatorException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::AssignmentOperatorException::AssignmentOperatorException(
    const AssignmentOperator* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: assignment operators only support value types\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
