/**
 * @file
 */
#include "bi/exception/ThisException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ThisException::ThisException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: 'this' used outside of class\n";
  msg = base.str();
}
