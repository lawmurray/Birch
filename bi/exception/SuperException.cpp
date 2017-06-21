/**
 * @file
 */
#include "bi/exception/SuperException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::SuperException::SuperException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: 'super' used outside of class\n";
  msg = base.str();
}
