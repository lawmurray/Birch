/**
 * @file
 */
#include "bi/exception/SuperBaseException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::SuperBaseException::SuperBaseException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: 'super' used outside of struct or class with a base type\n";
  msg = base.str();
}
