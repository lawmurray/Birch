/**
 * @file
 */
#include "bi/exception/NotAssignableException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::NotAssignableException::NotAssignableException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: left side of assignment is not assignable\n";
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: in\n";
  buf << expr << '\n';
  msg = base.str();
}
