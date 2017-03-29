/**
 * @file
 */
#include "InvalidAssignmentException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::InvalidAssignmentException::InvalidAssignmentException(
    const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: incompatible types in assignment\n";
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: in\n";
  buf << expr << '\n';
  msg = base.str();
}
